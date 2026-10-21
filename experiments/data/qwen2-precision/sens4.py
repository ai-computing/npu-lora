"""CPU emulation of an RNGD 'fp32' matmul that rounds its inputs to bf16 (forward and backward), applied to the
attention matmuls of the Qwen2 layer. Which reformulation keeps the LoRA gradients close to fp32?

usage: sens4.py REPO MODEL [layers] [S]
"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv.pop(1))
from npu_lora import llama as L  # noqa: E402

BF = torch.bfloat16
model = sys.argv[1]
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 2
S = int(sys.argv[3]) if len(sys.argv) > 3 else 128
path = L.resolve_model_path(model)
c = L.LlamaConfig.from_hf(path)
W = L.Weights(path)
Wl = [W.layer(i, c) for i in range(nl)]
norm, lm, emb = W.norm(c).float(), W.lm_head().float(), W.embed()
g = torch.Generator().manual_seed(1)
gen = torch.Generator().manual_seed(0)
master = []
for _ in range(nl):
    per = []
    for t in ("q", "v"):
        d_in, d_out = L.LORA_TARGETS[t](c)
        per += [((torch.rand(16, d_in, generator=gen) * 2 - 1) / math.sqrt(d_in)).to(BF).float(),
                torch.zeros(d_out, 16).normal_(0, 0.01, generator=g).to(BF).float()]
    master.append(per)
ids = torch.randint(0, c.vocab, (S,), generator=g)
labels = torch.roll(ids, -1)


def r16(x):
    return x.to(BF).float()


class MMB(torch.autograd.Function):
    """a @ b with both inputs rounded to bf16, fp32 accumulate/output; backward likewise"""
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return r16(a) @ r16(b)

    @staticmethod
    def backward(ctx, gy):
        a, b = ctx.saved_tensors
        return r16(gy) @ r16(b).transpose(-1, -2), r16(a).transpose(-1, -2) @ r16(gy)


def layer(h, w, lora, cos, sin, mask, rot, mode):
    """mode: 'fp32' (all fp32 exact) | 'emu' (attention matmuls bf16-input) | 'emu+center' | 'emu+center+split'"""
    mm = (lambda a, b: a @ b) if mode == "fp32" else MMB.apply
    x = L.rms(h, w["ln1"], c.eps)
    q = x @ w["wq"].t() + (x @ lora["q"][0].t()) @ lora["q"][1].t() * 2.0 + w["bq"]
    k = x @ w["wk"].t() + w["bk"]
    v = x @ w["wv"].t() + (x @ lora["v"][0].t()) @ lora["v"][1].t() * 2.0 + w["bv"]
    q, k = q.view(S, c.nh, c.hd), k.view(S, c.nkv, c.hd)
    cs, sn = cos.unsqueeze(1), sin.unsqueeze(1)
    if "split" in mode:            # rotate_half exactly: rot is a +-1 permutation, so hi and lo parts are exact
        rq = mm(r16(q), rot) + mm(q - r16(q), rot)
        rk = mm(r16(k), rot) + mm(k - r16(k), rot)
    else:
        rq, rk = mm(q, rot), mm(k, rot)
    q, k = (q * cs + rq * sn).transpose(0, 1), (k * cs + rk * sn).transpose(0, 1)
    if "center" in mode:           # softmax(q.(k - kbar)) == softmax(q.k): q_i.kbar is constant along the row
        k = k - k.mean(1, keepdim=True)
    v = v.view(S, c.nkv, c.hd).transpose(0, 1)
    rep = c.nh // c.nkv
    qg = q.reshape(c.nkv, rep * S, c.hd)
    att = mm(qg, k.transpose(-1, -2)).view(c.nkv, rep, S, S) / math.sqrt(c.hd) + mask
    att = torch.softmax(att, -1).view(c.nkv, rep * S, S)
    o = mm(att, v).view(c.nh, S, c.hd).transpose(0, 1).reshape(S, c.q_dim) @ w["wo"].t()
    h = h + o
    x = L.rms(h, w["ln2"], c.eps)
    return h + (F.silu(x @ w["wg"].t()) * (x @ w["wu"].t())) @ w["wd"].t()


def run(mode):
    ad = [[p.clone().requires_grad_(True) for p in per] for per in master]
    h = emb[ids].float()
    cos, sin = L.rope_tables(c, S, dtype=torch.float32)
    mask, rot = L.causal_mask(S), L.rotate_half_matrix(c.hd, dtype=torch.float32)
    for i in range(nl):
        w = {k_: v_.float() for k_, v_ in Wl[i].items()}
        h = layer(h, w, {"q": ad[i][:2], "v": ad[i][2:]}, cos, sin, mask, rot, mode)
    loss = F.cross_entropy(L.rms(h, norm, c.eps) @ lm.t(), labels)
    loss.backward()
    return [p.grad for per in ad for p in per]


ref = run("fp32")
for mode in ("emu", "emu+center", "emu+split", "emu+center+split"):
    gg = run(mode)
    print(mode, [round(float((a - b).norm() / b.norm()), 4) for a, b in zip(gg, ref)], flush=True)
