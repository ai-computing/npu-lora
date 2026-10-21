"""MoE 체크포인트의 MLP 절반(post-attention norm + MoE 블록)을 host 에서 고정한 채 돌린다.

furiosa-torch 에는 장치 쪽 gather 가 없어 token→expert 분배를 컴파일된 그래프 하나로 만들 수 없다. 그래서 MoE
레이어를 나눈다: attention + residual 은 장치 그래프(`llama.layer`, `LlamaConfig.moe`), 그 뒤 norm + MoE 블록은
여기서 HF 모듈 그대로(라우팅·shared expert·가중치 변환이 계열마다 HF 와 같다). expert 는 고정이고, 역전파는
입력(attention 출력)에 대한 기울기만 낸다 — LoRA 는 attention 투영에 붙는다.
"""
import time

import torch

BF = torch.bfloat16


class HostMLP:
    """레이어 i 의 (post_attention_layernorm, mlp) 를 HF 모델에서 가져와 host 에 둔다(bf16).

    forward(i, a)  = a + mlp(norm(a))            a: [S, H] host bf16 (attention + residual 출력)
    backward(i, a, g) = d forward(i, a) / d a · g  (순전파를 다시 계산한다)

    trainable (전체 파인튜닝): norm·router·expert(·shared expert) 도 학습한다. backward 가 파라미터 기울기를
    레이어별 fp32 누적기에 더하고, step() 이 fp32 master 에 host AdamW 를 적용해 bf16 모듈에 다시 쓴다.
    메모리는 파라미터당 bf16 2 B + master 4 B + AdamW 8 B (+ 누적 4 B) 다.
    """

    def __init__(self, path, dtype=BF, trainable=False, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        self.blocks = []
        for layer in model.model.layers:
            norm, mlp = layer.post_attention_layernorm, layer.mlp
            norm.requires_grad_(trainable)
            mlp.requires_grad_(trainable)
            self.blocks.append((norm, mlp))
        del model                      # embeddings, attention and lm_head are freed; the blocks stay
        self.trainable = trainable
        self.seconds = 0.0             # host time in forward/backward (step-time split: host MoE vs the rest)
        if trainable:
            self.params = [[p for p in (*norm.parameters(), *mlp.parameters())] for norm, mlp in self.blocks]
            self.master = [[p.detach().float().clone().requires_grad_(True) for p in ps] for ps in self.params]
            self.acc = [None] * len(self.blocks)
            kw = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
            mk = lambda ps: _adamw(ps, kw)
            self.opts = [mk(ms) for ms in self.master]

    def named_parameters(self, i):
        """(HF name, bf16 tensor) of layer i's host part"""
        norm, mlp = self.blocks[i]
        return ([(f"model.layers.{i}.post_attention_layernorm.{n}", p) for n, p in norm.named_parameters()]
                + [(f"model.layers.{i}.mlp.{n.replace('experts.inner.', 'experts.')}", p)
                   for n, p in mlp.named_parameters()])          # moe_device.HybridExperts wraps `experts`

    def _f(self, i, a):
        norm, mlp = self.blocks[i]
        y = mlp(norm(a.unsqueeze(0)))
        if isinstance(y, tuple):       # some MoE blocks also return router logits
            y = y[0]
        return a + y.reshape(a.shape).to(a.dtype)

    def forward(self, i, a):
        t0 = time.perf_counter()
        with torch.no_grad():
            y = self._f(i, a)
        self.seconds += time.perf_counter() - t0
        return y

    def backward(self, i, a, g):
        t0 = time.perf_counter()
        try:
            return self._backward(i, a, g)
        finally:
            self.seconds += time.perf_counter() - t0

    def _backward(self, i, a, g):
        x = a.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self._f(i, x)
            if not self.trainable:
                (gx,) = torch.autograd.grad(y, x, g.to(y.dtype))
                return gx
            gx, *gp = torch.autograd.grad(y, [x, *self.params[i]], g.to(y.dtype), allow_unused=True)
        gp = [torch.zeros_like(m) if q is None else q.float() for q, m in zip(gp, self.master[i])]
        self.acc[i] = gp if self.acc[i] is None else [a_ + b for a_, b in zip(self.acc[i], gp)]
        return gx

    def sq_norm(self):
        """sum of squares of the accumulated (not yet averaged) parameter gradients"""
        return sum(float(x.pow(2).sum()) for per in self.acc if per is not None for x in per)

    def step(self, clip, lr, k):
        """average over k micro-batches, clip, host AdamW on the fp32 masters, write back to the bf16 modules."""
        for i, per in enumerate(self.acc):
            if per is None:
                continue
            with torch.no_grad():
                for m, g in zip(self.master[i], per):
                    m.grad = g.div_(k).mul_(clip)
            for group in self.opts[i].param_groups:
                group["lr"] = lr
            self.opts[i].step()
            self.opts[i].zero_grad(set_to_none=True)
            with torch.no_grad():
                for p, m in zip(self.params[i], self.master[i]):
                    p.copy_(m.to(p.dtype))
            self.acc[i] = None


def _adamw(ps, kw):
    try:
        return torch.optim.AdamW(ps, fused=True, **kw)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.AdamW(ps, foreach=True, **kw)


class RoutingPins:
    """Record each MoE layer's top-k expert ids (`record`), or force recorded ones (`pin`), on the HostMLP routers.

    For comparing two backends on MoE checkpoints: tiny numeric differences flip some tokens' top-k choice, which
    changes their whole MoE output; pinned to the same experts, the comparison measures numerics only. The pinned
    router keeps its own probabilities (weights = its softmax at the pinned ids, renormalized like the router), so
    gradients still flow through the router."""

    def __init__(self, blocks):
        self.routers = [(i, mlp.gate) for i, (_, mlp) in enumerate(blocks) if hasattr(getattr(mlp, "gate", None), "top_k")]
        self.ids, self.mode, self._orig = {}, None, {}

    def __enter__(self):
        for i, r in self.routers:
            self._orig[i] = r.forward
            r.forward = (lambda h, i=i, r=r: self._route(i, r, h))
        return self

    def __exit__(self, *exc):
        for i, r in self.routers:
            r.forward = self._orig[i]

    def _route(self, i, router, hidden):
        probs, scores, ids = self._orig[i](hidden)
        if self.mode == "record":
            self.ids.setdefault(i, ids.detach().clone())          # the forward pass (backward recomputes it)
        elif self.mode == "pin" and i in self.ids:
            ids = self.ids[i]
            val = probs.gather(-1, ids)
            if getattr(router, "norm_topk_prob", True):
                val = val / val.sum(dim=-1, keepdim=True)
            scores = val.to(scores.dtype)
        return probs, scores, ids

    def flips(self, other):
        """per layer, tokens whose expert set differs between self's and other's recorded ids"""
        return {i: int(sum(set(a.tolist()) != set(b.tolist()) for a, b in zip(self.ids[i], other.ids[i])))
                for i in self.ids if i in other.ids}
