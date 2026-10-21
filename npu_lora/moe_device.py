"""MoE 학습에서 hot expert 를 장치에서 계산한다 (추론의 expert 배치와 같은 방식).

장치 쪽 gather 가 없으므로 라우팅·토큰 분배는 host 에서 한다. 한 레이어의 hot expert 들로 간 행을 host 에서 모아
expert 마다 같은 행 수(bucket, 가장 많이 받은 hot expert 기준)로 채운 [E_hot, bucket, H] 하나로 올리고, 쌓은
가중치 [E_hot, 2F, H]·[E_hot, H, F] 와 bmm 으로 계산한다 — 레이어당 순전파 한 번, 역전파 한 번의 장치 호출이다
(expert 마다 따로 부르면 호출·복사가 hot expert 수만큼 늘어 host 보다 느렸다, #35). 그래프는
(E_hot, H, F, bucket) 마다 한 번 컴파일한다. 역전파 그래프는 입력 기울기와(전체 파인튜닝이면) 쌓인 가중치의
기울기를 낸다. torch.autograd.Function 이 장치 역전파를 host autograd 에 잇고, 쌓인 가중치는
`gate_up_proj[hot]` 로 만들어 기울기가 expert 파라미터의 해당 조각에 그대로 들어간다.

cold expert 는 host 에서 expert 마다 계산한다. 어떤 expert 를 장치에 둘지는 라우팅 횟수로 정하고, step 사이에
다시 정한다(`replan`). 장치에는 레이어마다 지금 hot 인 expert 의 쌓인 가중치 한 벌만 둔다.

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
    """one expert (host path): x [n, H], gu [2F, H], dn [H, F] -> [n, H]"""
    g, u = (x @ gu.t()).chunk(2, dim=-1)
    return (F.silu(g) * u) @ dn.t()


def experts_fwd(x, gu, dn):
    """the hot experts of a layer at once: x [E, n, H], gu [E, 2F, H], dn [E, H, F] -> [E, n, H]"""
    g, u = (x @ gu.transpose(-1, -2)).chunk(2, dim=-1)
    return (F.silu(g) * u) @ dn.transpose(-1, -2)


def experts_bwd_fn(full):
    """(x, gu, dn, gy) -> gx (LoRA: experts frozen) or (gx, g_gu, g_dn) (full fine-tuning), batched like experts_fwd"""
    def bwd(x, gu, dn, gy):
        if full:
            _, pull = torch.func.vjp(experts_fwd, x, gu, dn)
            return pull(gy)
        _, pull = torch.func.vjp(lambda x_: experts_fwd(x_, gu, dn), x)
        return pull(gy)[0]
    return bwd


def bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return -(-n // BUCKETS[-1]) * BUCKETS[-1]


class DeviceExperts:
    """compiled batched expert graphs per (E_hot, H, F, bucket) and each layer's stacked hot-expert weights"""

    def __init__(self, backend, compile_parts, full, log=lambda m: None):
        self.be, self.cp, self.full, self.log = backend, compile_parts, full, log
        self.graphs = {}
        self.resident = {}                       # layer -> (hot expert ids, device gu [E,2F,H], device dn [E,H,F])
        self.calls = 0                           # device graph calls (forward + backward)

    def _graphs(self, E, H, Fd, n):
        key = (E, H, Fd, n)
        if key not in self.graphs:
            from . import graph as G
            x, gu, dn = torch.zeros(E, n, H, dtype=BF) + 0.01, torch.zeros(E, 2 * Fd, H, dtype=BF) + 0.01, \
                torch.zeros(E, H, Fd, dtype=BF) + 0.01
            gy = torch.zeros(E, n, H, dtype=BF) + 0.01
            k = f"moe-experts-E{E}-H{H}-F{Fd}-n{n}"
            # The first call comes from _HotBatch.forward, where autograd is off. Traced under no_grad, the vjp
            # keeps aten.silu_backward undecomposed, and the furiosa compiler rejects it ("Not importable for all
            # cases"); with grad mode on it is decomposed like in the dense layers.
            with torch.enable_grad():
                fwd = self.cp(f"{k}-fwd", G.build(experts_fwd, [x, gu, dn]), [x, gu, dn])
                bwd = self.cp(f"{k}-bwd{'-full' if self.full else ''}",
                              G.build(experts_bwd_fn(self.full), [x, gu, dn, gy]), [x, gu, dn, gy])
            self.graphs[key] = (fwd, bwd)
        return self.graphs[key]

    def weights(self, layer, hot, experts_module):
        """device (gu, dn) stacks of `hot` (sorted expert ids) of this layer; uploaded when the hot set changes"""
        r = self.resident.get(layer)
        if r is None or r[0] != hot:
            m, idx = experts_module, torch.tensor(hot)
            r = (hot, self.be.to_device(m.gate_up_proj.detach()[idx].to(BF).contiguous()),
                 self.be.to_device(m.down_proj.detach()[idx].to(BF).contiguous()))
            self.resident[layer] = r
        return r[1], r[2]

    def evict(self, keep):
        """keep: {(layer, expert)} of the new hot sets; drops the stacks whose hot set changed"""
        for layer in list(self.resident):
            if {(layer, e) for e in self.resident[layer][0]} != {k for k in keep if k[0] == layer}:
                del self.resident[layer]

    def refresh(self, layer, experts_module):
        """after a weight update (full fine-tuning) re-upload this layer's stacked hot experts"""
        r = self.resident.pop(layer, None)
        if r is not None:
            self.weights(layer, r[0], experts_module)


_NONE = torch.empty(0)


class _HotBatch(torch.autograd.Function):
    """all hot-expert rows of one layer in one device call each way.

    x [N, H] are the rows routed to hot experts (token copies), flat [N] each row's place in the packed
    [E_hot * nb, H] input (expert slot * nb + rank within the expert). gu, dn are the host stacks
    (m.gate_up_proj[hot], ...; empty when the experts are frozen): autograd routes their gradients back into
    the stacked parameters. The device uses its resident copies of the same stacks.
    """

    @staticmethod
    def forward(ctx, x, gu, dn, dev, layer, hot, flat, nb, m):
        E, H, Fd = len(hot), x.shape[1], m.down_proj.shape[-1]
        fwd, bwd = dev._graphs(E, H, Fd, nb)
        xp = torch.zeros(E * nb, H, dtype=BF)
        xp[flat] = x.detach().to(BF)
        dgu, ddn = dev.weights(layer, hot, m)
        y = dev.be.to_host(fwd(dev.be.to_device(xp.view(E, nb, H)), dgu, ddn)).reshape(E * nb, H)[flat]
        ctx.save_for_backward(xp, flat)
        ctx.dev, ctx.layer, ctx.hot, ctx.m, ctx.bwd, ctx.shape = dev, layer, hot, m, bwd, (E, nb, H)
        dev.calls += 1
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        xp, flat = ctx.saved_tensors
        dev, (E, nb, H) = ctx.dev, ctx.shape
        gyp = torch.zeros(E * nb, H, dtype=BF)
        gyp[flat] = gy.to(BF)
        dgu, ddn = dev.weights(ctx.layer, ctx.hot, ctx.m)
        out = ctx.bwd(dev.be.to_device(xp.view(E, nb, H)), dgu, ddn, dev.be.to_device(gyp.view(E, nb, H)))
        dev.calls += 1
        none = (None,) * 6
        if dev.full:
            gx, ggu, gdn = (dev.be.to_host(t) for t in out)
            return (gx.reshape(E * nb, H)[flat].to(gy.dtype), ggu.to(gy.dtype), gdn.to(gy.dtype), *none)
        return (dev.be.to_host(out).reshape(E * nb, H)[flat].to(gy.dtype), None, None, *none)


def pack(slot, n_hot):
    """slot [N] (hot-expert position of each row) -> (order, flat, nb): rows sorted by expert, each row's place in
    the packed [n_hot * nb, H] input, and the rows per expert nb (bucket of the busiest hot expert)"""
    order = torch.argsort(slot, stable=True)
    s = slot[order]
    n_per = torch.bincount(s, minlength=n_hot)
    rank = torch.arange(len(s)) - (torch.cumsum(n_per, 0) - n_per)[s]
    nb = bucket(int(n_per.max()))
    return order, s * nb + rank, nb


class HybridExperts(torch.nn.Module):
    """drop-in for a transformers >= 5 stacked experts module: hot experts on the device (one batched call per
    layer and direction), the rest on the host"""

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
        hot = tuple(sorted(self.hot))
        if hot:
            slot_of = torch.full((self.num_experts,), -1, dtype=torch.long)
            slot_of[list(hot)] = torch.arange(len(hot))
            slot = slot_of[flat]
            sel = (slot >= 0).nonzero().squeeze(1)
            if len(sel):
                order, place, nb = pack(slot[sel], len(hot))
                sel = sel[order]
                r, idx = rows[sel], torch.tensor(hot)
                # full fine-tuning: the host stacks carry the weight gradients back; LoRA: nothing to copy
                gu, dn = (m.gate_up_proj[idx], m.down_proj[idx]) if self.dev.full else (_NONE, _NONE)
                y = _HotBatch.apply(hidden[r], gu, dn, self.dev, self.layer, hot, place, nb, m)
                out = out.index_add(0, r, (y * wf[sel, None].to(y.dtype)).float())
        for e in torch.unique(flat).tolist():
            if e in self.hot:
                continue
            sel = (flat == e).nonzero().squeeze(1)
            r = rows[sel]
            y = expert_fwd(hidden[r], m.gate_up_proj[e], m.down_proj[e])
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
