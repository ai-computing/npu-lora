"""compile-only: batched hot-expert graphs (E_hot, H, F, bucket) and the attention graphs of a MoE checkpoint.
usage: python3 -m npu_lora.moe_precompile MODEL E_HOT BUCKETS(comma)"""
import json, sys, time
from npu_lora import backends, graph as G, moe_device as MD
from npu_lora.trainer import LoRAConfig, LoRATrainer, TrainConfig
model, E, bks = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
be = backends.FuriosaBackend(chip=0, cache_dir="/cache/furiosa", compile_only=True)
cfg = json.load(open(__import__("npu_lora.llama", fromlist=["x"]).resolve_model_path(model) + "/config.json"))
H, Fd = cfg["hidden_size"], cfg["moe_intermediate_size"]
cp = lambda k, gm, smp: G.compile_parts(be, k, gm, smp, 2, log=lambda m: print(m, flush=True))
dev = MD.DeviceExperts(be, cp, False)
for n in bks:
    t0 = time.time(); dev._graphs(E, H, Fd, n); print(json.dumps({"bucket": n, "seconds": round(time.time() - t0, 1)}), flush=True)
