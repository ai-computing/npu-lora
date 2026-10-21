"""ATOM rounding: results are rounded away from zero much more often than round-to-nearest-even would.

    python3 repro_rounding.py <device index>     # rebel-compiler + one free ATOM card

For each op the output is compared with a float64 truth computed from the same bf16 inputs.
  rel    ||out - truth|| / ||truth||
  bias   mean(err * sign(truth)) / mean(|err|): 0 for unbiased rounding, +1 if every error enlarges |out|
  ne_cpu fraction of elements that differ from torch's CPU result (round-to-nearest-even)
"""
import json
import sys

import torch
from torch.fx.experimental.proxy_tensor import make_fx

BF = torch.bfloat16
CASES = {
    "add_bf16": (lambda x, y, w: x + y, lambda x, y, w: x + y),
    "mul_bf16": (lambda x, y, w: x * y, lambda x, y, w: x * y),
    "mm_bf16": (lambda x, y, w: x @ w.t(), lambda x, y, w: x @ w.t()),
    "mm_fp32_out": (lambda x, y, w: x.float() @ w.float().t(), lambda x, y, w: x @ w.t()),
    "fp32_to_bf16": (lambda x, y, w: (x.float() * 1.0009765625 + y.float() * 0.0078125).to(BF),
                     lambda x, y, w: x * 1.0009765625 + y * 0.0078125),
    "add_fp32": (lambda x, y, w: x.float() + y.float() * 0.0078125, lambda x, y, w: x + y * 0.0078125),
}

if __name__ == "__main__":
    import rebel
    dev = int(sys.argv[1])
    print(json.dumps({"rebel-compiler": rebel.__version__, "torch": torch.__version__}))
    torch.manual_seed(0)
    args = [torch.randn(128, 2048).to(BF), (torch.randn(128, 2048) * 3).to(BF), (torch.randn(2048, 2048) * 0.02).to(BF)]
    info = [("x", [128, 2048], "bfloat16"), ("y", [128, 2048], "bfloat16"), ("w", [2048, 2048], "bfloat16")]
    for name, (f, truth_f) in CASES.items():
        gm = make_fx(f)(*args)
        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        keep = [i for i, n in enumerate(phs) if n.users]
        for n in phs:
            if not n.users:
                gm.graph.erase_node(n)
        gm.recompile()
        a = [args[i] for i in keep]
        rt = rebel.compile_from_torch(gm, [info[i] for i in keep]).create_runtime(device=dev, tensor_type="pt")
        out = rt(*a)
        out = (out[0] if isinstance(out, (list, tuple)) else out).double()
        cpu = gm(*a).double()
        truth = truth_f(*[t.double() for t in args])

        def stats(o):
            e = o - truth
            return {"rel": float(e.norm() / truth.norm()),
                    "bias": round(float((e * torch.sign(truth)).mean() / (e.abs().mean() + 1e-30)), 3)}
        print(json.dumps({"case": name, "atom": stats(out), "cpu": stats(cpu),
                          "ne_cpu": round(float((out != cpu).double().mean()), 4)}), flush=True)
        del rt
