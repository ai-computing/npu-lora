import json, statistics, time, torch
from npu_lora import backends, optim_device as O, llama as L
BF = torch.bfloat16
be = backends.make("furiosa:auto", cache_dir="/cache/furiosa")
W = L.Weights(L.resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"))
w0 = [W.get(f"model.layers.0.{L.HF_NAMES[k]}.weight", torch.float32).contiguous() for k in L.LAYER_WEIGHTS]
out = {}
def med(fn, n=7):
    fn(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return round(statistics.median(ts), 4)
# NPU: update + D2H of the last chunk output (forces completion) vs D2H alone
for dt in (torch.float32, BF):
    d = O.DeviceAdamW(be, dt, key="adamw")
    for s in {tuple(c.shape) for t in w0 for c in O.chunks(t)}:
        d.compile(s)
    st = [(be.to_device(c.to(dt).contiguous()), be.to_device(torch.zeros(c.shape, dtype=dt)), be.to_device(torch.zeros(c.shape, dtype=dt))) for t in w0 for c in O.chunks(t)]
    g = [be.to_device((torch.randn(c.shape) * 1e-4).to(BF).contiguous()) for t in w0 for c in O.chunks(t)]
    sc = [be.to_device(x) for x in O.scalars(1e-5, (0.9, 0.999), 1e-8, 0.01, 1)]
    last = {}
    def run():
        new, _ = d.step(st, g, sc); last["o"] = new
    def run_sync():
        run(); new = last["o"]; [x[0].cpu() for x in new[-1:]]
    tag = str(dt)[6:]
    out[f"npu_update_{tag}"] = med(run)
    out[f"npu_update_{tag}+d2h_last_chunk"] = med(run_sync)
    out[f"d2h_one_chunk_{tag}"] = med(lambda: st[-1][0].cpu())
    del st, g
# CPU fused AdamW vs thread count
for th in (4, 16, 48):
    torch.set_num_threads(th)
    ps = [t.clone().requires_grad_(True) for t in w0]
    for p in ps: p.grad = torch.randn_like(p) * 1e-4
    opt = torch.optim.AdamW(ps, lr=1e-5, weight_decay=0.01, fused=True)
    out[f"cpu_fused_adamw_{th}thr"] = med(opt.step)
print(json.dumps(out))
json.dump(out, open("/work/check_timing.json", "w"), indent=1)
