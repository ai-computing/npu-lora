"""MoE LoRA step time: host-only MoE vs hot experts on the device (issue #35). Same data, seed and recipe.

usage: python3 -m npu_lora.moe_bench MODEL DATA FRAC STEPS OUT.jsonl [backend]
"""
import json
import sys
import time

import torch

from npu_lora import backends
from npu_lora.data import PackedDataset
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig

model, data, frac, steps, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
spec = sys.argv[6] if len(sys.argv) > 6 else "furiosa:-1"
f = open(out, "a")


def log(**kw):
    kw["t"] = round(time.time(), 1)
    kw["load1"] = __import__("os").getloadavg()[0]
    f.write(json.dumps(kw) + "\n")
    f.flush()
    print(json.dumps(kw), flush=True)


ncomp = [0]


def tlog(s):
    d = json.loads(s)
    if d.get("event") == "compile_part":
        ncomp[0] += 1
    print(s, flush=True)


be = backends.make(spec, cache_dir="/cache/furiosa") if spec.startswith("furiosa") else backends.make(spec)
tc = TrainConfig(seq_len=128, lr=2e-4, steps=steps, accum=1, warmup=2, seed=0, moe_hot_fraction=frac)
tr = LoRATrainer(model, be, LoRAConfig(r=16, alpha=32, targets=("q", "v")), tc, log=tlog)
t0 = time.time()
tr.setup()
log(event="setup", frac=frac, seconds=round(time.time() - t0, 1), **be.describe())
from transformers import AutoTokenizer  # noqa: E402
ds = PackedDataset(data, AutoTokenizer.from_pretrained(tr.path), 128, seed=0)
for batch in ds.batches(1, steps):
    c0 = ncomp[0]
    calls0 = tr.moe_dev.calls if tr.moe_dev else 0
    m = tr.train_step(batch)
    st = tr.moe_stats()
    log(event="step", frac=frac, step=m["step"], loss=m["loss"], seconds=m["seconds"], compiles=ncomp[0] - c0,
        device_calls=(st.get("device_calls", 0) - calls0), hot=sum(st.get("hot_per_layer", [])),
        host_moe_s=m.get("host_moe_s"), device_and_transfer_s=m.get("device_and_transfer_s"))
