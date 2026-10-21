"""#35: profile a MoE LoRA step and compare configurations round-robin in one process (one model load).

Configs: host-only / hot experts on the device, each with HostMLP.keep_graph off (backward recomputes the forward)
and on. Per step it logs the split: attention+head on the device (incl. transfers), HostMLP forward, backward
recompute, backward autograd, and inside them the hot-expert device calls, the cold experts' forward, the router.
usage: python3 -m npu_lora.moe_prof MODEL DATA FRAC SEQ ROUNDS OUT.jsonl
"""
import json
import os
import sys
import time

import torch

from npu_lora import backends, moe as M, moe_device as MD
from npu_lora.data import PackedDataset
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig

model, data, frac, S, rounds, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
f = open(out, "a")
T = {}


def add(k, dt):
    T[k] = T.get(k, 0.0) + dt


def timed(k, fn):
    def w(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            add(k, time.perf_counter() - t0)
    return w


def log(**kw):
    kw.update(t=round(time.time(), 1), load1=os.getloadavg()[0] if hasattr(os, "getloadavg") else None)
    f.write(json.dumps(kw) + "\n")
    f.flush()
    print(json.dumps(kw), flush=True)


# timers (wrapping only; no behaviour change)
M.HostMLP.forward = timed("mlp_fwd", M.HostMLP.forward)
_orig_f = M.HostMLP._f
_in_bwd = [False]


def _f(self, i, a):
    t0 = time.perf_counter()
    try:
        return _orig_f(self, i, a)
    finally:
        if _in_bwd[0]:
            add("bwd_recompute", time.perf_counter() - t0)


M.HostMLP._f = _f
_orig_bwd = M.HostMLP._backward


def _bwd(self, i, a, g):
    _in_bwd[0] = True
    try:
        return _orig_bwd(self, i, a, g)
    finally:
        _in_bwd[0] = False


M.HostMLP._backward = timed("mlp_bwd", _bwd)
MD._HotBatch.forward = staticmethod(timed("hot_dev_fwd", MD._HotBatch.forward))
MD._HotBatch.backward = staticmethod(timed("hot_dev_bwd", MD._HotBatch.backward))
MD.expert_fwd = timed("cold_fwd", MD.expert_fwd)

spec = os.environ.get("PROF_BACKEND", "furiosa:-1")
be = backends.make(spec, cache_dir="/cache/furiosa") if spec.startswith("furiosa") else backends.make(spec)
tc = TrainConfig(seq_len=S, lr=2e-4, steps=200, accum=1, warmup=2, seed=0, moe_hot_fraction=frac,
                 moe_replan_every=10 ** 9)
tr = LoRATrainer(model, be, LoRAConfig(r=16, alpha=32, targets=("q", "v")), tc, log=lambda s: None)
t0 = time.time()
tr.setup()
for norm, mlp in tr.host_mlp.blocks:
    mlp.gate.forward = timed("router_fwd", mlp.gate.forward)
log(event="setup", seconds=round(time.time() - t0, 1), seq=S, **be.describe())
from transformers import AutoTokenizer  # noqa: E402
if os.environ.get("PROF_RANDOM_VOCAB"):                # smoke test on a tiny random model
    _V = int(os.environ["PROF_RANDOM_VOCAB"])
    it = iter([[(x, torch.roll(x, -1), None)] for x in (torch.randint(0, _V, (S,)) for _ in range(200))])
else:
    it = iter(PackedDataset(data, AutoTokenizer.from_pretrained(tr.path), S, seed=0).batches(1, 200))


def step(name, hot, keep):
    tr.host_mlp.keep_graph = keep
    for w, h in zip(tr.moe_wraps, plan):
        if w is not None:
            w.hot = set(h) if hot else set()
    T.clear()
    s0 = tr.moe_stats()
    mlp0 = tr.host_mlp.seconds
    m = tr.train_step(next(it))
    s1 = tr.moe_stats()
    mlp_s = tr.host_mlp.seconds - mlp0
    log(event="step", cfg=name, step=m["step"], loss=m["loss"], seconds=m["seconds"],
        device_attn_head_s=round(m["seconds"] - mlp_s, 3), **{k: round(v, 3) for k, v in T.items()},
        device_calls=s1["device_calls"] - s0["device_calls"], uploads=s1.get("weight_uploads", 0) - s0.get("weight_uploads", 0))


plan = [set() for _ in tr.moe_wraps]
for _ in range(4):                                    # warm-up host-only: routing counts for the hot set
    step("warmup", False, True)
MD.replan(tr.moe_wraps, tr.moe_dev)
plan = [set(w.hot) if w is not None else None for w in tr.moe_wraps]
cfgs = [("host", False, False), ("host+keep", False, True), ("hot", True, False), ("hot+keep", True, True)]
for r in range(rounds):                               # round-robin: load drift hits every config alike
    for name, hot, keep in (cfgs if r % 2 == 0 else cfgs[::-1]):
        step(name, hot, keep)
