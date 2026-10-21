"""MoE LoRA step time in one process (one model load): host-only, hot experts with a fixed hot set, hot experts
re-planned every step, host-only again. usage: python3 -m npu_lora.moe_bench2 MODEL DATA FRAC OUT.jsonl"""
import json, os, sys, time
from npu_lora import backends, moe_device as MD
from npu_lora.data import PackedDataset
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig
model, data, frac, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
f = open(out, "a")
def log(**kw):
    kw.update(t=round(time.time(), 1), load1=os.getloadavg()[0]); f.write(json.dumps(kw) + "\n"); f.flush(); print(json.dumps(kw), flush=True)
be = backends.make("furiosa:-1", cache_dir="/cache/furiosa")
tc = TrainConfig(seq_len=128, lr=2e-4, steps=60, accum=1, warmup=2, seed=0, moe_hot_fraction=frac, moe_replan_every=10 ** 9)
tr = LoRATrainer(model, be, LoRAConfig(r=16, alpha=32, targets=("q", "v")), tc, log=lambda s: None)
t0 = time.time(); tr.setup(); log(event="setup", seconds=round(time.time() - t0, 1), **be.describe())
from transformers import AutoTokenizer
it = iter(PackedDataset(data, AutoTokenizer.from_pretrained(tr.path), 128, seed=0).batches(1, 60))
def phase(name, n, hot=None, replan=10 ** 9):
    tr.tc.moe_replan_every = replan
    if hot == "none":
        for w in tr.moe_wraps:
            if w is not None: w.hot = set()
    elif hot == "plan":
        MD.replan(tr.moe_wraps, tr.moe_dev)
    for _ in range(n):
        s0 = tr.moe_stats()
        m = tr.train_step(next(it))
        s1 = tr.moe_stats()
        log(event="step", phase=name, step=m["step"], loss=m["loss"], seconds=m["seconds"], host_moe_s=m.get("host_moe_s"),
            device_calls=s1["device_calls"] - s0["device_calls"], uploads=s1.get("weight_uploads", 0) - s0.get("weight_uploads", 0),
            upload_s=round(s1.get("weight_upload_s", 0) - s0.get("weight_upload_s", 0), 3), hot=sum(s1["hot_per_layer"]))
phase("host", 4, "none")                     # warm-up and routing counts for the plan
MD.replan(tr.moe_wraps, tr.moe_dev)
plan = [set(w.hot) if w is not None else None for w in tr.moe_wraps]
for i in range(int(sys.argv[5]) if len(sys.argv) > 5 else 12):   # alternate step by step: load drift cancels
    if i % 2 == 0:
        for w, h in zip(tr.moe_wraps, plan):
            if w is not None: w.hot = set(h)
        phase("hot", 1)
    else:
        phase("host", 1, "none")
