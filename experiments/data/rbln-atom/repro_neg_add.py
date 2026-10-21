"""rebel-compiler 0.11: aten.add(aten.neg(x), 1.0) does not compile (DEVICE_GRAPH_CONVERSION).

In the Llama-3.2-1B SiLU-backward part (the pair, then mm [128, 8192] @ [8192, 2048]) the compiler process
segfaults instead (exit -11); with the small shapes below it reports DEVICE_GRAPH_CONVERSION.

    python3 repro_neg_add.py            # needs rebel-compiler and an ATOM card on the host (compile only, no run)

Each graph is built with torch.fx from aten ops and compiled with rebel.compile_from_torch in a child process,
so a segfault shows up as exit code -11 instead of killing the script.
"""
import os
import subprocess
import sys
import tempfile

import torch
import torch.fx as fx

aten = torch.ops.aten


def graph(body, shapes, dtype="bfloat16"):
    g = fx.Graph()
    xs = [g.placeholder(f"x{i}") for i in range(len(shapes))]
    g.output(body(g, *xs))
    return fx.GraphModule(torch.nn.Module(), g), [(f"x{i}", list(s), dtype) for i, s in enumerate(shapes)]


def call(g, op, *args):
    return g.call_function(op, args)


CASES = {
    # 1 - x written as neg + add.Scalar (what a functionalized SiLU backward can contain)
    "neg_add_scalar": (lambda g, x: call(g, aten.add.Scalar, call(g, aten.neg.default, x), 1.0), [(4, 64)]),
    # the same value as rsub: compiles
    "rsub_scalar": (lambda g, x: call(g, aten.rsub.Scalar, x, 1.0), [(4, 64)]),
    # each op alone: compiles
    "neg": (lambda g, x: call(g, aten.neg.default, x), [(4, 64)]),
    "add_scalar": (lambda g, x: call(g, aten.add.Scalar, x, 1.0), [(4, 64)]),
    # neg + add.Scalar followed by a matmul (the real part shapes segfault the compiler)
    "neg_add_scalar_mm": (lambda g, x, w: call(g, aten.mm.default, call(g, aten.mul.Tensor, x, call(
        g, aten.add.Scalar, call(g, aten.neg.default, x), 1.0)), w), [(128, 256), (256, 64)]),
    "rsub_scalar_mm": (lambda g, x, w: call(g, aten.mm.default, call(g, aten.mul.Tensor, x, call(
        g, aten.rsub.Scalar, x, 1.0)), w), [(128, 256), (256, 64)]),
}

CHILD = ("import sys, torch, rebel\n"
         "o = torch.load(sys.argv[1], weights_only=False)\n"
         "rebel.compile_from_torch(o['gm'], o['info']).save(sys.argv[2])\n")

if __name__ == "__main__":
    import rebel
    print("rebel-compiler", rebel.__version__, "torch", torch.__version__)
    d = tempfile.mkdtemp()
    for name, (body, shapes) in CASES.items():
        for dtype in ("bfloat16", "float32"):
            gm, info = graph(body, shapes, dtype)
            src = os.path.join(d, "g.pt")
            torch.save({"gm": gm, "info": info}, src)
            r = subprocess.run([sys.executable, "-c", CHILD, src, os.path.join(d, "g.rbln")],
                               capture_output=True, text=True)
            err = [l.strip() for l in (r.stderr + r.stdout).splitlines() if "Error" in l or "Segmentation" in l]
            print(f"{name:18s} {dtype:8s} exit {r.returncode:4d}  {err[-1][-160:] if r.returncode and err else ''}")
