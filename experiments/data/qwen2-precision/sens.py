"""CPU: how far are the bf16 graph gradients (what `npu-lora check` compares) from an fp32 reference?

usage: sens.py MODEL [layers] [seq]
"""
import json
import sys
import time

import torch

sys.path.insert(0, sys.argv.pop(1))
from npu_lora import backends, llama as L          # noqa: E402
from npu_lora.trainer import LoRATrainer, LoRAConfig, TrainConfig  # noqa: E402

model = sys.argv[1]
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 2
S = int(sys.argv[3]) if len(sys.argv) > 3 else 128


def mk():
    tr = LoRATrainer(model, backends.CPUBackend(), LoRAConfig(), TrainConfig(seq_len=S), log=lambda s: None, max_mm=0)
    tr.c = tr.c.__class__(**{**tr.c.__dict__, "n_layers": nl})
    tr.setup()
    g = torch.Generator().manual_seed(1)
    for per in tr.master:
        for b in per[1::2]:
            b.detach().normal_(0, 0.01, generator=g)
    tr._upload_adapters()
    ids = torch.randint(0, tr.c.vocab, (S,), generator=g)
    return tr, ids, torch.roll(ids, -1)


def ref(tr, ids, labels, dt=torch.float32, logits_dt=torch.float32):
    """autograd reference with every tensor in dt"""
    c = tr.c
    ad = [[p.detach().to(BF).to(dt).requires_grad_(True) for p in per] for per in tr.master]
    h = tr.embed[ids].to(dt)
    cos, sin = L.rope_tables(c, S, dtype=dt)
    mask = L.causal_mask(S)
    rot = L.rotate_half_matrix(c.hd, dtype=dt)
    for i in range(c.n_layers):
        w = {k: v.to(dt) for k, v in tr.W.layer(i, c).items()}
        lo = {t: (ad[i][2 * j], ad[i][2 * j + 1]) for j, t in enumerate(tr.lora.targets)}
        h = L.layer(c, h, w, lo, tr.lora.scale, cos, sin, mask, rot)
    x = L.rms(h, tr.W.norm(c).to(dt), c.eps)
    logits = (x.to(logits_dt) @ tr.W.lm_head().to(logits_dt).t()).float()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return float(loss), [[p.grad.float() for p in per] for per in ad]


def cmp(ga, gb):
    rows = []
    for i, (x, y) in enumerate(zip(ga, gb)):
        for j, (p, q) in enumerate(zip(x, y)):
            rows.append((i, j, round(float((p - q).norm() / (q.norm() + 1e-12)), 4),
                         round(float(torch.nn.functional.cosine_similarity(p.flatten(), q.flatten(), dim=0)), 5)))
    return rows


BF = torch.bfloat16
tr, ids, labels = mk()
t0 = time.time()
lb, gb, _ = tr.forward_backward(ids, labels)
print("graph bf16 loss", lb, round(time.time() - t0, 1), "s")
l32, g32 = ref(tr, ids, labels)
print("fp32 ref loss", l32)
print("bf16 graph vs fp32:", cmp(gb, g32))
l16, g16 = ref(tr, ids, labels, BF, torch.float32)
print("bf16 eager (fp32 logits) loss", l16, "vs fp32:", cmp(g16, g32))
print("bf16 graph vs bf16 eager:", cmp(gb, g16))
