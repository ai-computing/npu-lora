"""#40 repro: furiosa-torch 2026.2's device memory type is process-global, so an upload from one thread
while another thread runs a compiled graph can land in DRAM_SHARED (0 bytes on a single-chip RNGD) and fail.

  PYTHONPATH=<npu-train checkout> python3 race_repro.py flag      # mode set in a thread is seen by the main thread
  PYTHONPATH=<npu-train checkout> python3 race_repro.py watch     # graph calls flip the mode for a moment
  PYTHONPATH=<npu-train checkout> python3 race_repro.py race      # OOM on DRAM_SHARED within seconds
  PYTHONPATH=<npu-train checkout> python3 race_repro.py fixed     # npu_lora.backends (lock): no OOM in 120 s

Uses one card (furiosa:auto) and a trivial compiled graph (x * 2, [4096, 4096] fp32).
"""
import sys
import threading
import time

import torch

from npu_lora import backends, graph as G

mode = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120
be = backends.make("furiosa:auto")
from furiosa.native_torch import memory as M  # noqa: E402  (after furiosa.torch)

if mode == "flag":
    ev, go = threading.Event(), threading.Event()

    def other():
        prev = M._set_dram_shared_mode()
        ev.set()
        go.wait()
        M._set_memory_type(prev)

    t = threading.Thread(target=other)
    t.start()
    ev.wait()
    print("main thread sees dram_shared mode set by another thread:", M._is_dram_shared_mode())
    try:
        torch.zeros(1024, 4096, dtype=torch.bfloat16).to(be.dev)
        print("upload ok")
    except Exception as e:  # noqa: BLE001
        print("upload failed:", repr(e))
    go.set()
    t.join()
    sys.exit(0)

smp = [torch.zeros(4096, 4096)]
fn = G.compile_parts(be, "repro-x2-4096", G.build(lambda x: (x * 2.0,), smp), smp, max_mm=0, log=print)
x = be.to_device(torch.ones(4096, 4096))
if mode != "fixed":
    fn = fn.fn if isinstance(fn, backends.Serialized) else fn      # call the module without the lock
stop, err, n_up = [False], [], [0]

if mode == "watch":
    hits, polls = [0], [0]

    def poll():
        while not stop[0]:
            polls[0] += 1
            hits[0] += M._is_dram_shared_mode()

    t = threading.Thread(target=poll)
    t.start()
    for _ in range(300):
        fn(x)
    stop[0] = True
    t.join()
    print(f"{hits[0]} of {polls[0]} polls saw dram_shared mode during 300 graph calls")
    sys.exit(0)

host = torch.zeros(1024, 4096, dtype=torch.bfloat16)
up = be.to_device if mode == "fixed" else (lambda t: t.to(be.dev))


def uploader():
    try:
        while not stop[0]:
            up(host)
            n_up[0] += 1
    except Exception as e:  # noqa: BLE001
        err.append(repr(e)[:200])


ths = [threading.Thread(target=uploader) for _ in range(4)]
for t in ths:
    t.start()
t0, calls = time.time(), 0
while time.time() - t0 < secs and not err:
    fn(x)
    calls += 1
stop[0] = True
for t in ths:
    t.join()
print(f"{mode}: {calls} graph calls, {n_up[0]} uploads in {time.time() - t0:.1f} s, errors: {err[:2]}")
