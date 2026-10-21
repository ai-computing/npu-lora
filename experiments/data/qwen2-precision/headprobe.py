"""RNGD: which head-chunk statistic loses precision? (issue #33, negative loss)

One vocab chunk [S, Vc] with confident rows (label = row argmax, logits ~ 10-40). Variants of the chunk-statistics
graph run on RNGD; each is compared with float64 truth computed from the same bf16 inputs.
usage: python3 -m npu_lora.headprobe OUT.json
"""
import json
import sys

import torch

from npu_lora import backends, graph as G

BF = torch.bfloat16
S, H, Vc = 128, 1536, 4096


def v_bf16(x, lm, iota, lab):
    lg = (x @ lm.t()).float()
    m = lg.amax(-1, keepdim=True)
    return m, torch.exp(lg - m).sum(-1, keepdim=True), (lg * (iota == lab).float()).sum(-1, keepdim=True)


def v_fp32(x, lm, iota, lab):
    lg = x.float() @ lm.float().t()
    m = lg.amax(-1, keepdim=True)
    return m, torch.exp(lg - m).sum(-1, keepdim=True), (lg * (iota == lab).float()).sum(-1, keepdim=True)


def v_fp32_mround(x, lm, iota, lab):
    lg = x.float() @ lm.float().t()
    m = lg.amax(-1, keepdim=True).to(BF).float()
    return m, torch.exp(lg - m).sum(-1, keepdim=True), (lg * (iota == lab).float()).sum(-1, keepdim=True)


def v_bf16_mround(x, lm, iota, lab):
    lg = (x @ lm.t()).float()
    m = lg.amax(-1, keepdim=True).to(BF).float()
    return m, torch.exp(lg - m).sum(-1, keepdim=True), (lg * (iota == lab).float()).sum(-1, keepdim=True)


VARIANTS = {"bf16": v_bf16, "fp32": v_fp32, "fp32_mround": v_fp32_mround, "bf16_mround": v_bf16_mround}


def main():
    out = sys.argv[1]
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(S, H, generator=g)).to(BF)
    lm = (torch.randn(Vc, H, generator=g) * 0.02).to(BF)
    # make rows confident: plant a direction of row i's x into a label row
    lab_ids = torch.randint(0, Vc, (S,), generator=g)
    lm[lab_ids] = (x.float() * (torch.rand(S, 1, generator=g) * 0.02 + 0.005)).to(BF)
    iota = torch.arange(Vc, dtype=torch.float32).view(1, -1)
    lab = lab_ids.float().view(-1, 1)
    lg64 = x.double() @ lm.double().t()
    m64 = lg64.amax(-1, keepdim=True)
    lse64 = m64 + torch.log(torch.exp(lg64 - m64).sum(-1, keepdim=True))
    t64 = lg64.gather(1, lab_ids.view(-1, 1))
    res = {"truth": {"ce_min": float((lse64 - t64).min()), "ce_mean": float((lse64 - t64).mean()),
                     "logit_absmax": float(lg64.abs().max())}}
    be = backends.make("furiosa:-1", cache_dir="/cache/furiosa")
    args = [x, lm, iota, lab]
    for name, fn in VARIANTS.items():
        gm = G.build(fn, args)
        f = G.compile_parts(be, f"headprobe-{name}", gm, args, 2)
        m, s, t = (be.to_host(v).double() for v in f(*[be.to_device(a.contiguous()) for a in args]))
        lse = m + torch.log(s)
        res[name] = {"m_err_max": float((m - m64).abs().max()), "t_err_max": float((t - t64).abs().max()),
                     "lse_err_max": float((lse - lse64).abs().max()), "ce_min": float((lse - t).min()),
                     "neg_rows": int(((lse - t) < 0).sum()), "t_minus_m_max": float((t - m).max())}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
