"""AdamW 갱신을 장치 그래프로 돌린다(실험: NPUTRAIN-34).

입력 (w, m, v, g, lr, β1, β2, eps, wd, 1−β1^t, 1−β2^t) → 출력 (w', m', v'[, bf16 w']).
스칼라는 [1,1] 텐서로 넣고(0차원은 컴파일러 패닉), step 에 따라 바뀌는 1−β^t 도 입력이라 그래프 하나를 모든
step 에 쓴다. 식은 torch.optim.AdamW(fused) 와 같다:
    w ← w·(1 − lr·wd);  m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²
    w ← w − (lr / (1−β1^t)) · m / (√v / √(1−β2^t) + eps)

레이어 가중치는 [CHUNK_ROWS, CHUNK_COLS] 조각으로 나눠 같은 그래프 하나로 돈다(norm 은 [1, hidden] 그래프).
두 변형: fp32(상태·master 모두 fp32, 다음 순전파용 bf16 w' 도 낸다), bf16(w·m·v 모두 bf16, 계산은 fp32).

  python -m npu_train.optim_device --backend furiosa:auto --model meta-llama/Llama-3.1-8B-Instruct --out r.json
"""
import argparse
import json
import statistics
import time

import torch

from . import graph as G
from . import llama as L

BF = torch.bfloat16
CHUNK_ROWS, CHUNK_COLS = 1024, 4096


def adamw_fn(w, m, v, g, lr, b1, b2, eps, wd, bc1, bc2):
    """한 조각의 AdamW. w·m·v 의 dtype 으로 돌려준다. w 가 fp32 면 bf16 w' 도 낸다."""
    wf, mf, vf, gf = w.float(), m.float(), v.float(), g.float()
    # 1 − x 를 rsub 로 쓰지 않는다(-x + 1) — 상수 텐서를 만들지 않게
    m2 = mf * b1 + gf * (-b1 + 1)
    v2 = vf * b2 + gf * gf * (-b2 + 1)
    denom = torch.sqrt(v2) / torch.sqrt(bc2) + eps
    w2 = wf * (-(lr * wd) + 1) - (lr / bc1) * (m2 / denom)
    out = (w2.to(w.dtype), m2.to(m.dtype), v2.to(v.dtype))
    return out + (w2.to(BF),) if w.dtype == torch.float32 else out


def scalars(lr, betas, eps, wd, step):
    b1, b2 = betas
    vals = (lr, b1, b2, eps, wd, 1 - b1 ** step, 1 - b2 ** step)
    return [torch.tensor([[x]], dtype=torch.float32) for x in vals]


def chunks(t):
    """레이어 가중치 하나 → 연속 조각 리스트(view). norm 은 [1, hidden] 하나."""
    if t.dim() == 1:
        return [t.view(1, -1)]
    return list(t.reshape(-1, CHUNK_ROWS, CHUNK_COLS).unbind(0))


def unchunk(parts, like):
    return torch.cat([p.reshape(-1) for p in parts]).view(like.shape)


class DeviceAdamW:
    """조각 모양·dtype 마다 컴파일한 갱신 그래프로 레이어 하나를 갱신한다. 상태는 장치 텐서 조각."""

    def __init__(self, backend, dtype, log=print, key="adamw"):
        self.be, self.dtype, self.log, self.key = backend, dtype, log, key
        self.fns, self.gms, self.compile_s = {}, {}, {}

    def graph(self, shape):
        k = tuple(shape)
        if k not in self.gms:
            sample = self.sample(shape)
            self.gms[k] = G.build(adamw_fn, sample)
        return self.gms[k]

    def sample(self, shape):
        # 0/상수 샘플은 값 문제를 못 잡는다 — 정확도는 compare_parts 와 연쇄 비교로 따로 본다
        gen = torch.Generator().manual_seed(0)
        r = lambda dt: (torch.randn(shape, generator=gen) * 0.02).to(dt)
        return [r(self.dtype), r(self.dtype), r(self.dtype).abs() * 1e-4, r(BF)] + scalars(1e-5, (0.9, 0.999), 1e-8, 0.01, 1)

    def compile(self, shape):
        k = tuple(shape)
        if k not in self.fns:
            t0 = time.time()
            name = f"{self.key}-{str(self.dtype).replace('torch.', '')}-{'x'.join(map(str, k))}"
            self.fns[k] = G.compile_parts(self.be, name, self.graph(shape), self.sample(shape), max_mm=0, log=self.log)
            self.compile_s[k] = time.time() - t0
        return self.fns[k]

    def step(self, state, grads, sc):
        """state: [(w, m, v)] 장치 조각, grads: 장치 조각, sc: 장치 스칼라 → 새 state, bf16 w'(fp32 일 때)."""
        new, wbf = [], []
        for (w, m, v), g in zip(state, grads):
            out = self.fns[tuple(w.shape)](w, m, v, g, *sc)
            out = tuple(out)
            new.append(out[:3])
            if len(out) > 3:
                wbf.append(out[3])
        return new, wbf


def adamw_scaled_fn(w, m, v, g, gscale, lr, b1, b2, eps, wd, bc1, bc2):
    """adamw_fn with the gradient scaled on the device by gscale = clip / accum ([1,1] input)."""
    return adamw_fn(w, m, v, g.float() * gscale, lr, b1, b2, eps, wd, bc1, bc2)


def resident_shapes(W, c):
    """The distinct 2-D weight shapes of a decoder layer: one ResidentAdamW graph each (1-D norms stay on the host).
    Read from the safetensors header, so no weights are loaded (Llama-3.1-8B: 4 shapes)."""
    return sorted({s for s in (W.shape(f"model.layers.0.{L.hf_name(k)}") for k in L.layer_weights(c)) if len(s) >= 2})


class ResidentAdamW:
    """AdamW for weights whose fp32 master and moments stay on the device (trainer `resident_layers`).

    One graph per whole weight shape — no chunking, so the bf16 w' output is the next forward's weight as is.
    Inputs (w32, m, v, g bf16, gscale, lr, β1, β2, eps, wd, 1−β1^t, 1−β2^t) → (w32', m', v', w' bf16).
    """

    def __init__(self, backend, log=print, key="adamw-res"):
        self.be, self.log, self.key = backend, log, key
        self.fns = {}

    @staticmethod
    def sample(shape):
        gen = torch.Generator().manual_seed(0)
        r = lambda: torch.randn(shape, generator=gen) * 0.02
        return [r(), r(), r().abs() * 1e-4, r().to(BF), torch.tensor([[0.5]])] + scalars(1e-5, (0.9, 0.999), 1e-8, 0.0, 1)

    def compile(self, shape):
        k = tuple(shape)
        if k not in self.fns:
            smp = self.sample(k)
            self.fns[k] = G.compile_parts(self.be, f"{self.key}-fp32-{'x'.join(map(str, k))}",
                                          G.build(adamw_scaled_fn, smp), smp, max_mm=0, log=self.log)
        return self.fns[k]

    def step(self, state, g, gscale, sc):
        w, m, v = state
        return tuple(self.compile(tuple(w.shape))(w, m, v, g, gscale, *sc))


# ---------------------------------------------------------------- 실험

def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def cat(ts):
    return torch.cat([t.reshape(-1).float() for t in ts])


def median_time(fn, reps, sync):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="furiosa:auto")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad-scale", type=float, default=1e-4)
    ap.add_argument("--cache-dir")
    ap.add_argument("--out", default="adamw-device.json")
    a = ap.parse_args(argv)
    from . import backends, diagnose
    be = backends.make(a.backend, **({"cache_dir": a.cache_dir} if a.cache_dir and a.backend.startswith(("furiosa", "rngd")) else {}))
    res = {"backend": be.describe(), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "args": vars(a), "chunk": [CHUNK_ROWS, CHUNK_COLS]}
    log = lambda *x: print(*x, flush=True)

    def save():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    W = L.Weights(L.resolve_model_path(a.model))
    names = [f"model.layers.{a.layer}.{L.HF_NAMES[k]}.weight" for k in L.LAYER_WEIGHTS]
    w0 = [W.get(n, torch.float32).contiguous() for n in names]
    n_par = sum(t.numel() for t in w0)
    res["layer_params"] = n_par
    log(f"[adamw] layer {a.layer}: {n_par / 1e6:.1f}M params, shapes {[tuple(t.shape) for t in w0]}")
    betas, eps = (0.9, 0.999), 1e-8

    # 기울기: 원소마다 고정 신호 + 잡음(같은 seed → 모든 변형이 같은 bf16 기울기)
    gen = torch.Generator().manual_seed(1)
    mu = [torch.randn(t.shape, generator=gen) * a.grad_scale * 0.5 for t in w0]

    def grads(step):
        gg = torch.Generator().manual_seed(1000 + step)
        return [(m_ + torch.randn(m_.shape, generator=gg) * a.grad_scale).to(BF) for m_ in mu]

    # ---- 1. 컴파일 + 조각별 비교(compare_parts)
    opt_dev = {}
    for dt in (torch.float32, BF):
        tag = str(dt).replace("torch.", "")
        d = DeviceAdamW(be, dt, log=log, key="adamw")
        res.setdefault("compile", {})[tag] = {}
        for shape in sorted({tuple(c.shape) for t in w0 for c in chunks(t)}):
            gm = d.graph(shape)
            entry = res["compile"][tag].setdefault("x".join(map(str, shape)), {})
            entry["ops"] = [str(n.target).replace("aten.", "") for n in gm.graph.nodes if n.op == "call_function"]
            try:
                entry["compare_parts"] = diagnose.compare_parts(be, f"adamw-cmp-{tag}", gm, d.sample(shape), log=log)
                d.compile(shape)
                entry["compile_s"] = round(d.compile_s[shape], 1)
                log(f"[adamw] {tag} {shape}: compiled {entry['compile_s']}s")
            except Exception as e:  # noqa: BLE001 — 막힌 노드를 남기고 끝낸다
                entry["error"] = repr(e)[:2000]
                log(f"[adamw] {tag} {shape}: compile FAILED {e!r}; searching first failing node")
                hit = diagnose.first_failing_node(be, gm, d.sample(shape), log=log)
                entry["first_failing_node"] = diagnose.describe(hit[1]) if hit else None
                save()
                raise
        opt_dev[tag] = d
    save()

    # ---- 2. 연쇄 정확도: CPU fused AdamW(fp32) 기준, 장치 fp32 / 장치 bf16 / CPU 에서 같은 그래프 bf16
    ref = [t.clone().requires_grad_(True) for t in w0]
    try:
        opt = torch.optim.AdamW(ref, lr=a.lr, betas=betas, eps=eps, weight_decay=a.wd, fused=True)
        res["cpu_adamw"] = "fused"
    except (RuntimeError, TypeError, ValueError):
        opt = torch.optim.AdamW(ref, lr=a.lr, betas=betas, eps=eps, weight_decay=a.wd, foreach=True)
        res["cpu_adamw"] = "foreach"
    up = be.to_device

    def init_state(dt):
        return [[(up(c.to(dt).contiguous()), up(torch.zeros(c.shape, dtype=dt)), up(torch.zeros(c.shape, dtype=dt)))
                 for c in chunks(t)] for t in w0]

    dev_state = {"float32": init_state(torch.float32), "bfloat16": init_state(BF)}
    emu_state = [[(c.to(BF).contiguous(), torch.zeros(c.shape, dtype=BF), torch.zeros(c.shape, dtype=BF))
                  for c in chunks(t)] for t in w0]
    emu_gm = opt_dev["bfloat16"].gms
    w_bf_prev = [t.to(BF) for t in w0]
    rows = []
    for step in range(1, a.steps + 1):
        g = grads(step)
        w_prev = [p.detach().clone() for p in ref]
        for p, gi in zip(ref, g):
            p.grad = gi.float()
        opt.step()
        sc_host = scalars(a.lr, betas, eps, a.wd, step)
        sc = [up(s) for s in sc_host]
        row = {"step": step}
        for tag, d in opt_dev.items():
            new_layer, wbf_layer = [], []
            for st, gi in zip(dev_state[tag], g):
                gd = [up(c.contiguous()) for c in chunks(gi)]
                s2, wbf = d.step(st, gd, sc)
                new_layer.append(s2)
                wbf_layer.append(wbf)
            be.sync()
            dev_state[tag] = new_layer
            host = lambda k: [unchunk([be.to_host(s[k]) for s in st], t) for st, t in zip(new_layer, w0)]
            w_d, m_d, v_d = host(0), host(1), host(2)
            m_r = [opt.state[p]["exp_avg"] for p in ref]
            v_r = [opt.state[p]["exp_avg_sq"] for p in ref]
            w_r = [p.detach() for p in ref]
            dw_r = cat(w_r) - cat(w0)
            dw_d = cat(w_d) - cat(w0)
            r = {"w": rel(cat(w_d), cat(w_r)), "m": rel(cat(m_d), cat(m_r)), "v": rel(cat(v_d), cat(v_r)),
                 "dw_cum": rel(dw_d, dw_r), "per_tensor_dw_cum": [rel(x.float() - y, z.detach() - y) for x, y, z in zip(w_d, w0, w_r)]}
            if tag == "float32":
                wbf = [unchunk([be.to_host(c) for c in cs], t) for cs, t in zip(wbf_layer, w0)]
                r["w_bf16_out_vs_cast"] = rel(cat(wbf), cat([x.to(BF) for x in w_d]))
            else:
                # bf16 해상도보다 작은 갱신: 이번 step 의 기준 갱신량/가중치 < 2^-8(bf16 반 ulp 상한)이면 사라질 수 있다
                step_r = cat(w_r) - cat(w_prev)
                base = cat(w_prev).abs()
                nz = step_r != 0
                r["frac_update_below_2^-8"] = float(((step_r.abs() / base.clamp_min(1e-30)) < 2 ** -8)[nz].float().mean())
                r["frac_update_below_2^-9"] = float(((step_r.abs() / base.clamp_min(1e-30)) < 2 ** -9)[nz].float().mean())
                w_bf_now = cat(w_d)
                r["frac_bf16_w_unchanged"] = float((w_bf_now == cat(w_bf_prev))[nz].float().mean())
                w_bf_prev = [x.clone() for x in w_d]
            row[tag] = r
        # CPU 에서 같은 bf16 그래프(장치 오차와 bf16 형식 오차를 나눠 본다)
        new_emu = []
        for st, gi in zip(emu_state, g):
            new_emu.append([tuple(emu_gm[tuple(w_.shape)](w_, m_, v_, gc.contiguous(), *sc_host)[:3])
                            for (w_, m_, v_), gc in zip(st, chunks(gi))])
        emu_state = new_emu
        w_e = [unchunk([s[0] for s in st], t) for st, t in zip(emu_state, w0)]
        w_dbf = [unchunk([be.to_host(s[0]) for s in st], t) for st, t in zip(dev_state["bfloat16"], w0)]
        row["bf16_cpu_graph"] = {"dw_cum": rel(cat(w_e) - cat(w0), cat([p.detach() for p in ref]) - cat(w0)),
                                 "dev_vs_cpu_graph_w": rel(cat(w_dbf), cat(w_e))}
        rows.append(row)
        log(f"[adamw] step {step}: fp32 dw_cum {row['float32']['dw_cum']:.3g} w {row['float32']['w']:.3g} | "
            f"bf16 dw_cum {row['bfloat16']['dw_cum']:.3g} below2^-8 {row['bfloat16']['frac_update_below_2^-8']:.3f} "
            f"unchanged {row['bfloat16']['frac_bf16_w_unchanged']:.3f} | bf16 cpu-graph dw_cum {row['bf16_cpu_graph']['dw_cum']:.3g} "
            f"dev-vs-cpu {row['bf16_cpu_graph']['dev_vs_cpu_graph_w']:.3g}")
        res["chain"] = rows
        save()
    del emu_state

    # ---- 3. 시간(레이어 하나, 중앙값)
    T = {}
    sc = [up(s) for s in scalars(a.lr, betas, eps, a.wd, 1)]
    for tag, d in opt_dev.items():
        g_dev = [[up(c.contiguous()) for c in chunks(gi)] for gi in grads(1)]
        st = dev_state[tag]
        T[f"npu_update_{tag}"], _ = median_time(lambda: [d.step(s, gd, sc) for s, gd in zip(st, g_dev)], a.reps, be.sync)
        del g_dev
    for p, gi in zip(ref, grads(1)):
        p.grad = gi.float()
    T["cpu_fused_adamw"], _ = median_time(opt.step, a.reps, lambda: None)

    def h2d(ts):
        return [up(t) for t in ts]

    for tag, dt in (("fp32", torch.float32), ("bf16", BF)):
        host_state = [c.to(dt).contiguous() for t in w0 for c in chunks(t)] * 3          # w, m, v
        dev = h2d(host_state)
        be.sync()
        bufs = [torch.empty_like(t) for t in host_state]
        T[f"state_h2d_{tag}"], _ = median_time(lambda: h2d(host_state), a.reps, be.sync)
        T[f"state_d2h_{tag}"], _ = median_time(lambda: [b.copy_(x) for b, x in zip(bufs, dev)], a.reps, be.sync)
        res[f"state_bytes_{tag}"] = sum(t.numel() * t.element_size() for t in host_state)
        del dev, bufs, host_state
    # 지금 방식: 기울기 bf16 내리기(상주 host 버퍼) + bf16 가중치 올리기
    g_host = [gi.contiguous() for gi in grads(1)]
    g_dev = h2d(g_host)
    gbufs = [torch.empty_like(t) for t in g_host]
    w_bf = [t.to(BF).contiguous() for t in w0]
    be.sync()
    T["grad_d2h_bf16"], _ = median_time(lambda: [b.copy_(x) for b, x in zip(gbufs, g_dev)], a.reps, be.sync)
    T["weight_h2d_bf16"], _ = median_time(lambda: h2d(w_bf), a.reps, be.sync)
    T["grad_d2h_bf16_tcpu"], _ = median_time(lambda: [x.cpu() for x in g_dev], a.reps, be.sync)
    res["grad_bytes_bf16"] = sum(t.numel() * 2 for t in g_host)
    res["time_s"] = {k: round(v, 4) for k, v in T.items()}
    res["per_layer_s"] = {
        "cpu_path": round(T["grad_d2h_bf16"] + T["cpu_fused_adamw"] + T["weight_h2d_bf16"], 4),
        "npu_stream_fp32": round(T["state_h2d_fp32"] + T["npu_update_float32"] + T["state_d2h_fp32"], 4),
        "npu_stream_bf16": round(T["state_h2d_bf16"] + T["npu_update_bfloat16"] + T["state_d2h_bf16"], 4),
    }
    log("[adamw] time_s " + json.dumps(res["time_s"]))
    log("[adamw] per_layer_s " + json.dumps(res["per_layer_s"]))
    save()


if __name__ == "__main__":
    main()
