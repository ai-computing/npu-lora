"""RNGD: run the layer forward graph's parts one by one (cached compiles), print each before it runs.
usage: python3 -m npu_lora.runparts MODEL"""
import sys, time, torch
from npu_lora import backends, diagnose, graph as G
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig
from torch.fx.passes.split_module import split_module
model = sys.argv[1]
be = backends.make("furiosa:-1", cache_dir="/cache/furiosa")
tr = LoRATrainer(model, backends.CPUBackend(), LoRAConfig(), TrainConfig(seq_len=128), log=lambda s: None, max_mm=0)
c, lc = tr.c, tr.lora
h, w, ad, cos, sin, mask, rot = tr._samples()
g = torch.Generator().manual_seed(0)
h = torch.randn(128, c.H, generator=g).to(torch.bfloat16)
w = [x + torch.randn(x.shape, generator=g).to(x.dtype) * 0.02 for x in w]
args = [h, *w, *ad, cos, sin, mask, rot]
key = f"llama-v2-H{c.H}-I{c.I}-nh{c.nh}-kv{c.nkv}{c.key}-S128-r{lc.r}-{''.join(lc.targets)}-s{lc.scale}"
for name, fn, a in (("fwd", G.layer_fwd_fn(c, lc.targets, lc.scale), args),
                    ("bwd", G.layer_bwd_fn(c, lc.targets, lc.scale), args + [h + 0.01])):
    gm = G.build(fn, a)
    diagnose.compare_parts(be, f"{key}-{name}", gm, a, 2, log=lambda m: print(time.strftime("%X"), m, flush=True))
