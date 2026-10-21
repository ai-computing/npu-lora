"""RNGD micro-benchmark of one batched hot-expert call (Qwen3-30B-A3B shapes) vs the host per-expert path.
usage: python3 -m npu_lora.moe_micro OUT.json"""
import json
import sys
import time

import torch

from npu_lora import backends, graph as G, moe_device as MD

BF = torch.bfloat16
H, Fd, E = 2048, 768, 32
be = backends.make("furiosa:-1", cache_dir="/cache/furiosa")
cp = lambda k, gm, smp: G.compile_parts(be, k, gm, smp, 2, log=lambda m: None)
dev = MD.DeviceExperts(be, cp, False)
g = torch.Generator().manual_seed(0)
gu = (torch.randn(E, 2 * Fd, H, generator=g) * 0.02).to(BF)
dn = (torch.randn(E, H, Fd, generator=g) * 0.02).to(BF)
dgu, ddn = be.to_device(gu), be.to_device(dn)
res = {"threads": torch.get_num_threads()}


def tm(f, n=5):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


for nb in (16, 32, 64, 128):
    fwd, bwd = dev._graphs(E, H, Fd, nb)
    x = (torch.randn(E, nb, H, generator=g)).to(BF)
    gy = (torch.randn(E, nb, H, generator=g) * 1e-3).to(BF)
    xd, gyd = be.to_device(x), be.to_device(gy)
    r = {"upload_s": tm(lambda: be.to_device(x)),
         "fwd_s": tm(lambda: be.to_host(fwd(xd, dgu, ddn))),
         "bwd_s": tm(lambda: be.to_host(bwd(xd, dgu, ddn, gyd))),
         "download_s": tm(lambda: be.to_host(xd))}
    # host: the same rows per expert, forward + input gradient (LoRA: experts frozen)
    def host():
        for e in range(E):
            xe = x[e].clone().requires_grad_(True)
            y = MD.expert_fwd(xe, gu[e], dn[e])
            torch.autograd.grad(y, xe, gy[e])
    r["host_fwd_bwd_s"] = tm(host, 2)
    res[nb] = r
    print(nb, json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
