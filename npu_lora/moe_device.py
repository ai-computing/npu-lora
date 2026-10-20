"""MoE 학습에서 hot expert 를 장치에서 계산한다 (추론의 expert 배치와 같은 방식).

장치 쪽 gather 가 없으므로 라우팅·토큰 분배는 host 에서 한다. expert e 로 간 행만 host 에서 모아 정해진 행 수
(bucket)로 채워 올리고, (H, F, bucket) 마다 한 번 컴파일한 expert 순전파/역전파 그래프로 계산한다. 역전파 그래프는
입력 기울기와(전체 파인튜닝이면) 그 expert 의 가중치 기울기를 낸다. torch.autograd.Function 이 장치 역전파를
host autograd 에 잇기 때문에, 가중치 기울기는 쌓인 expert 파라미터의 해당 조각에 그대로 들어간다.

cold expert 는 host 에서 같은 식으로 계산한다. 어떤 expert 를 장치에 둘지는 라우팅 횟수로 정하고, step 사이에
다시 정한다(`replan`).

지원: 표준 SwiGLU expert (gate_up_proj [E, 2F, H], down_proj [E, H, F], bias 없음, 기본 gate) — Qwen3-MoE,
Qwen2-MoE(routed experts), Mixtral. 그 밖의 배치는 모두 host 에서 돈다.
"""
import torch
import torch.nn.functional as F

BF = torch.bfloat16
BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024)


def standard_swiglu(experts):
    """the stacked SwiGLU layout the device graphs implement"""
    try:
        from transformers.integrations.moe import _default_apply_gate
        default_gate = getattr(type(experts), "_apply_gate", _default_apply_gate) is _default_apply_gate
    except Exception:
        default_gate = not hasattr(type(experts), "_apply_gate")
    return (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
            and not getattr(experts, "is_transposed", False) and not getattr(experts, "has_bias", False)
            and default_gate and getattr(experts, "act_fn", None) is not None
            and type(experts.act_fn).__name__ in ("SiLU", "SiLUActivation"))


def expert_fwd(x, gu, dn):
    """x [n, H], gu [2F, H], dn [H, F] -> [n, H]"""
    g, u = (x @ gu.t()).chunk(2, dim=-1)
    return (F.silu(g) * u) @ dn.t()


def expert_bwd_fn(full):
    """(x, gu, dn, gy) -> gx (LoRA: experts frozen) or (gx, g_gu, g_dn) (full fine-tuning)"""
    def bwd(x, gu, dn, gy):
        if full:
            _, pull = torch.func.vjp(expert_fwd, x, gu, dn)
            return pull(gy)
        _, pull = torch.func.vjp(lambda x_: expert_fwd(x_, gu, dn), x)
        return pull(gy)[0]
    return bwd


def bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return -(-n // BUCKETS[-1]) * BUCKETS[-1]


class DeviceExperts:
    """compiled expert graphs per (H, F, bucket) and the device copies of the hot experts' weights"""

    def __init__(self, backend, compile_parts, full, log=lambda m: None):
        self.be, self.cp, self.full, self.log = backend, compile_parts, full, log
        self.graphs = {}
        self.resident = {}                       # (layer, expert) -> (device gu, device dn)
        self.calls = 0

    def _graphs(self, H, Fd, n):
        key = (H, Fd, n)
        if key not in self.graphs:
            from . import graph as G
            x, gu, dn = torch.zeros(n, H, dtype=BF) + 0.01, torch.zeros(2 * Fd, H, dtype=BF) + 0.01, \
                torch.zeros(H, Fd, dtype=BF) + 0.01
            gy = torch.zeros(n, H, dtype=BF) + 0.01
            k = f"moe-expert-H{H}-F{Fd}-n{n}"
            fwd = self.cp(f"{k}-fwd", G.build(expert_fwd, [x, gu, dn]), [x, gu, dn])
            bwd = self.cp(f"{k}-bwd{'-full' if self.full else ''}",
                          G.build(expert_bwd_fn(self.full), [x, gu, dn, gy]), [x, gu, dn, gy])
            self.graphs[key] = (fwd, bwd)
        return self.graphs[key]

    def weights(self, layer, e, gu, dn):
        k = (layer, e)
        if k not in self.resident:
            self.resident[k] = (self.be.to_device(gu.detach().to(BF).contiguous()),
                                self.be.to_device(dn.detach().to(BF).contiguous()))
        return self.resident[k]

    def evict(self, keep):
        for k in [k for k in self.resident if k not in keep]:
            del self.resident[k]

    def refresh(self, layer, experts_module):
        """after a weight update (full fine-tuning) re-upload the resident experts of this layer"""
        for (l, e) in [k for k in self.resident if k[0] == layer]:
            self.resident[(l, e)] = (self.be.to_device(experts_module.gate_up_proj[e].detach().to(BF).contiguous()),
                                     self.be.to_device(experts_module.down_proj[e].detach().to(BF).contiguous()))


class _OnDevice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gu, dn, dev, layer, e):
        n = x.shape[0]
        nb = bucket(n)
        H, Fd = x.shape[1], dn.shape[1]
        fwd, bwd = dev._graphs(H, Fd, nb)
        xp = torch.zeros(nb, H, dtype=BF)
        xp[:n] = x.detach().to(BF)
        dgu, ddn = dev.weights(layer, e, gu, dn)
        xd = dev.be.to_device(xp)
        y = dev.be.to_host(fwd(xd, dgu, ddn))[:n].to(x.dtype)
        ctx.save_for_backward(xp)
        ctx.n, ctx.dev, ctx.key, ctx.bwd = n, dev, (layer, e), bwd
        dev.calls += 1
        return y

    @staticmethod
    def backward(ctx, gy):
        (xp,) = ctx.saved_tensors
        dev, n = ctx.dev, ctx.n
        gyp = torch.zeros(xp.shape, dtype=BF)
        gyp[:n] = gy.to(BF)
        dgu, ddn = dev.weights(*ctx.key, None, None)
        out = ctx.bwd(dev.be.to_device(xp), dgu, ddn, dev.be.to_device(gyp))
        if dev.full:
            gx, ggu, gdn = (dev.be.to_host(t) for t in out)
            return gx[:n].to(gy.dtype), ggu.to(gy.dtype), gdn.to(gy.dtype), None, None, None
        return dev.be.to_host(out)[:n].to(gy.dtype), None, None, None, None, None


class HybridExperts(torch.nn.Module):
    """drop-in for a transformers >= 5 stacked experts module: hot experts on the device, the rest on the host"""

    def __init__(self, inner, layer, dev):
        super().__init__()
        self.inner, self.layer, self.dev = inner, layer, dev
        self.hot = set()
        self.counts = torch.zeros(inner.num_experts, dtype=torch.long)

    @property
    def num_experts(self):
        return self.inner.num_experts

    def forward(self, hidden, top_k_index, top_k_weights):
        T, H = hidden.shape
        k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        rows = torch.arange(T).repeat_interleave(k)
        wf = top_k_weights.reshape(-1)
        self.counts += torch.bincount(flat.detach().cpu(), minlength=self.num_experts)[:self.num_experts]
        m = self.inner
        out = torch.zeros(T, H, dtype=torch.float32)
        for e in torch.unique(flat).tolist():
            sel = (flat == e).nonzero().squeeze(1)
            r = rows[sel]
            x = hidden[r]
            if e in self.hot:
                y = _OnDevice.apply(x, m.gate_up_proj[e], m.down_proj[e], self.dev, self.layer, e)
            else:
                y = expert_fwd(x, m.gate_up_proj[e], m.down_proj[e])
            out = out.index_add(0, r, (y * wf[sel, None].to(y.dtype)).float())
        return out.to(hidden.dtype)


def attach(blocks, dev, hot_fraction):
    """replace each layer's standard experts module by a HybridExperts; returns the wrappers (None where the
    layout is not supported). The first hot set is empty: `replan` fills it from routing counts."""
    wraps = []
    for i, (_, mlp) in enumerate(blocks):
        ex = getattr(mlp, "experts", None)
        if ex is None or not standard_swiglu(ex):
            wraps.append(None)
            continue
        w = HybridExperts(ex, i, dev)
        mlp.experts = w
        wraps.append(w)
    dev.hot_fraction = hot_fraction
    return wraps


def replan(wraps, dev):
    """hot set per layer = the most routed experts so far (fraction dev.hot_fraction); evict the rest"""
    keep = set()
    for w in wraps:
        if w is None:
            continue
        n = int(round(w.num_experts * dev.hot_fraction))
        w.hot = set(torch.argsort(w.counts, descending=True)[:n].tolist()) if n else set()
        keep |= {(w.layer, e) for e in w.hot}
    dev.evict(keep)
