"""Issue #33 on RNGD: RNGD vs CPU-bf16 vs fp32 reference gradients (the `npu-lora check` setup), then
diagnose.compare_parts on layer 0's forward/backward graphs and the head chunks with the real inputs.

usage: python3 -m npu_lora.diag33 MODEL OUT.json [layers] [seq]
"""
import json
import sys
import time

import torch

from npu_lora import backends, diagnose, graph as G, llama as L
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig

model, out = sys.argv[1], sys.argv[2]
nl = int(sys.argv[3]) if len(sys.argv) > 3 else 2
S = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BF = torch.bfloat16
res = {"model": model, "layers": nl, "seq": S}


def say(**kw):
    print(json.dumps(kw), flush=True)


def mk(be, max_mm):
    tr = LoRATrainer(model, be, LoRAConfig(), TrainConfig(seq_len=S), log=lambda s: None, max_mm=max_mm)
    tr.c = tr.c.__class__(**{**tr.c.__dict__, "n_layers": nl})
    tr.setup()
    g = torch.Generator().manual_seed(1)
    for per in tr.master:
        for b in per[1::2]:
            b.detach().normal_(0, 0.01, generator=g)
    tr._upload_adapters()
    ids = torch.randint(0, tr.c.vocab, (S,), generator=g)
    return tr, ids, torch.roll(ids, -1)


def ref32(tr, ids, labels):
    c = tr.c
    ad = [[p.detach().to(BF).float().requires_grad_(True) for p in per] for per in tr.master]
    h = tr.embed[ids].float()
    cos, sin = L.rope_tables(c, S, dtype=torch.float32)
    mask, rot = L.causal_mask(S), L.rotate_half_matrix(c.hd, dtype=torch.float32)
    for i in range(c.n_layers):
        w = {k: v.float() for k, v in tr.W.layer(i, c).items()}
        lo = {t: (ad[i][2 * j], ad[i][2 * j + 1]) for j, t in enumerate(tr.lora.targets)}
        h = L.layer(c, h, w, lo, tr.lora.scale, cos, sin, mask, rot)
    logits = L.rms(h, tr.W.norm(c).float(), c.eps) @ tr.W.lm_head().float().t()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return float(loss), [[p.grad.float() for p in per] for per in ad]


def cmp(a, b):
    return [[round(float((x - y).norm() / (y.norm() + 1e-12)), 4) for x, y in zip(p, q)] for p, q in zip(a, b)]


import os
spec = os.environ.get("DIAG_BACKEND", "furiosa:-1")
npu = backends.make(spec, cache_dir="/cache/furiosa") if spec.startswith("furiosa") else backends.make(spec)
cpu = backends.CPUBackend()
t0 = time.time()
trn, ids, labels = mk(npu, 2)
say(event="npu_setup", seconds=round(time.time() - t0, 1), **npu.describe())
trc, _, _ = mk(cpu, 0)
cap = {}
fwd0, bwd0 = trc.g_fwd, trc.g_bwd
trc.g_fwd = lambda *a: cap.setdefault("fwd", a) and fwd0(*a)
trc.g_bwd = lambda *a: (cap.__setitem__("bwd", a), bwd0(*a))[1]   # the last call is layer 0
ln, gn, _ = trn.forward_backward(ids, labels)
lc, gc, _ = trc.forward_backward(ids, labels)
l32, g32 = ref32(trc, ids, labels)
res["loss"] = {"npu": ln, "cpu_bf16": lc, "fp32": l32}
res["grad_rel"] = {"npu_vs_cpu_bf16": cmp(gn, gc), "npu_vs_fp32": cmp(gn, g32), "cpu_bf16_vs_fp32": cmp(gc, g32)}
say(**res)

# head: per-row (m, s, t) of the vocab chunks on both backends, labels = the CPU argmax (the most confident
# rows: CE = lse - t must be >= 0; a negative value on one backend means t and lse disagree inside the chunk graph)
def head_rows(tr, h, lab):
    be = tr.be_head
    x = tr.g_rms(be.to_device(h), tr.dev_norm)
    st = []
    for lm_j, bias_j in zip(tr.dev_lm, tr.dev_bias):
        m, s_ = tr.g_stats(x, lm_j, bias_j)
        st.append(tuple(be.to_host(v).float() for v in (m, s_)))
    lse = L.ce_combine(st)
    t = L.target_logits(be.to_host(x), tr.lm_host, lab)
    return lse.view(-1), t.view(-1), torch.cat([v[0] for v in st], -1)


with torch.no_grad():
    mask = L.causal_mask(S)
    h = trc.embed[ids]
    for i in range(nl):
        h = trc.g_fwd(h, *trc.dev_W[i], *trc.dev_AD[i], *trc._cs(i), mask, trc.dev_rot)
    logits = L.rms(h, trc.W.norm(trc.c), trc.c.eps).float() @ trc.W.lm_head().float().t()
    am = logits.argmax(-1)
    head = {}
    for name, tr in (("npu", trn), ("cpu", trc)):
        lse, t, m = head_rows(tr, h, am)
        head[name] = (lse, t, m)
    (ln_, tn_, mn_), (lc_2, tc_2, mc_2) = head["npu"], head["cpu"]
    res["head"] = {"ce_min_npu": float((ln_ - tn_).min()), "ce_min_cpu": float((lc_2 - tc_2).min()),
                   "neg_rows_npu": int(((ln_ - tn_) < 0).sum()), "neg_rows_cpu": int(((lc_2 - tc_2) < 0).sum()),
                   "ce_mean_npu": float((ln_ - tn_).mean()), "ce_mean_cpu": float((lc_2 - tc_2).mean()),
                   "t_absdiff_max": float((tn_ - tc_2).abs().max()), "lse_absdiff_max": float((ln_ - lc_2).abs().max()),
                   "m_absdiff_max": float((mn_ - mc_2).abs().max()),
                   "t_minus_rowmax_npu_max": float((tn_ - mn_.max(-1).values).max())}
    say(head=res["head"])

# compare_parts with the real layer-0 inputs (graphs are the trainer's, so the compiled parts come from the cache)
c, lc_ = trc.c, trc.lora
key = f"llama-v2-H{c.H}-I{c.I}-nh{c.nh}-kv{c.nkv}{c.key}-S{S}-r{lc_.r}-{''.join(lc_.targets)}-s{lc_.scale}"
parts = {}
for name, fn in (("fwd", G.layer_fwd_fn(c, lc_.targets, lc_.scale)), ("bwd", G.layer_bwd_fn(c, lc_.targets, lc_.scale))):
    args = [a.contiguous() for a in cap[name]]
    gm = G.build(fn, args)
    rows = diagnose.compare_parts(npu, f"{key}-{name}", gm, args, 2, log=lambda m: print(m, flush=True))
    parts[name] = rows
res["parts"] = parts
json.dump(res, open(out, "w"), indent=1)
say(event="done", out=out)
