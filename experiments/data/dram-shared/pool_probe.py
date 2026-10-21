"""#40 probe: what lives in DRAM vs DRAM_SHARED on RNGD, and how big each pool is.

usage: PYTHONPATH=<npu-train checkout> python3 pool_probe.py <mode>
  info     furiosa-torch memory API / env strings
  upload   keep uploading 64 MB bf16 tensors (to_device) until OOM
  graph    keep graph outputs (x*2, [4096,4096] fp32 = 64 MB) until OOM
  mix N        hold N GiB of uploads, then fill with graph outputs (one pool or two?)
  graphfirst   fill with graph outputs, then upload
"""
import glob
import os
import re
import subprocess
import sys
import time

import torch

mode = sys.argv[1]
BF = torch.bfloat16


def say(*a):
    print(*a, flush=True)


if mode == "info":
    import furiosa.torch as ft
    say("furiosa.torch attrs:", [x for x in dir(ft) if re.search("mem|alloc|stat|cache|pool|dram", x, re.I)])
    for name in ("furiosa", "rngd"):
        m = getattr(torch, name, None)
        if m is not None:
            say(f"torch.{name}:", [x for x in dir(m) if re.search("mem|alloc|stat|cache|pool", x, re.I)])
    base = os.path.dirname(ft.__file__)
    say("pkg dir", base)
    sos = [p for p in glob.glob(os.path.join(os.path.dirname(base), "**", "*.so*"), recursive=True)
           if "furiosa" in p]
    for p in sos:
        say("so", p, os.path.getsize(p))
    pat = re.compile(rb"[ -~]{4,}")
    seen = set()
    for p in sos:
        data = open(p, "rb").read()
        for s in pat.findall(data):
            t = s.decode()
            if re.search(r"DRAM_SHARED|dram.?shared|shared.?dram|SharedDram|DramShared", t, re.I) or \
               re.search(r"^(FURIOSA|RNGD|NPU|FURIOSA_TORCH|NPU_)[A-Z0-9_]{3,}$", t):
                if t not in seen and len(t) < 300:
                    seen.add(t)
                    say("str", os.path.basename(p), t)
    sys.exit(0)

from npu_lora import backends, graph as G  # noqa: E402

be = backends.make("furiosa:auto", cache_dir="/cache/furiosa")
dev = be.dev
say("device", dev, be.describe())
for k in ("memory_allocated", "memory_reserved", "memory_stats", "mem_get_info"):
    for m in (getattr(torch, "furiosa", None), getattr(torch, "rngd", None)):
        f = getattr(m, k, None) if m is not None else None
        if f:
            try:
                say("api", k, f())
            except Exception as e:  # noqa: BLE001
                say("api", k, "error", repr(e)[:200])
MB = 1 << 20
keep = []


def fill_uploads(limit_gb=None, shape=(4096, 8192)):
    t = torch.zeros(shape, dtype=BF)
    n = 0
    t0 = time.time()
    try:
        while limit_gb is None or n * t.numel() * 2 < limit_gb * (1 << 30):
            keep.append(be.to_device(t))
            n += 1
            if n % 50 == 0:
                say(f"upload {n} x {t.numel() * 2 // MB} MB = {n * t.numel() * 2 / (1 << 30):.2f} GiB {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        say(f"upload OOM after {n} x {t.numel() * 2 // MB} MB = {n * t.numel() * 2 / (1 << 30):.2f} GiB: {repr(e)[:200]}")
        return n, True
    return n, False


def graph_fn():
    smp = [torch.zeros(4096, 4096)]
    return G.compile_parts(be, "probe-x2-4096", G.build(lambda x: (x * 2.0,), smp), smp, max_mm=0, log=say)


def fill_graph(fn, x):
    outs, n = [], 0
    try:
        while True:
            out = fn(x)
            outs.append(out[0] if isinstance(out, (tuple, list)) else out)
            n += 1
            if n % 50 == 0:
                say(f"graph {n} x 64 MB = {n / 16:.2f} GiB")
    except Exception as e:  # noqa: BLE001
        say(f"graph OOM after {n} x 64 MB = {n / 16:.2f} GiB: {repr(e)[:200]}")
    return outs


if mode == "upload":
    fill_uploads()
elif mode == "graph":
    fn = graph_fn()
    x = be.to_device(torch.ones(4096, 4096))
    fill_graph(fn, x)
elif mode == "mix":
    # DRAM_SHARED first, then graph outputs: are they the same pool?
    fn = graph_fn()
    x = be.to_device(torch.ones(4096, 4096))
    lim = float(sys.argv[2]) if len(sys.argv) > 2 else 8
    n, oom = fill_uploads(lim)
    say(f"holding {n} uploads ({lim} GiB)")
    outs = fill_graph(fn, x)
    say("graph outputs kept", len(outs))
    # free graph outputs, try uploads again
    outs = None
    n2, _ = fill_uploads()
    say("uploads after freeing graph outputs", n2)
elif mode == "graphfirst":
    fn = graph_fn()
    x = be.to_device(torch.ones(4096, 4096))
    outs = fill_graph(fn, x)
    say("holding graph outputs", len(outs))
    n2, _ = fill_uploads()
    outs = outs[: len(outs) // 2]
    say("freed half the graph outputs")
    n3, _ = fill_uploads()
say("DONE", mode)
