"""Compile only every N-th graph part (shard K of N) so several pods fill the compile cache in parallel.

  SHARD=K NSHARD=N python3 w7_shard_compile.py compile --model M ... (same args as `npu-lora compile`)
Parts already in the cache are loaded as usual; skipped parts return a stub that must never run
(compile-only mode does not run graphs). The real run afterwards finds every part in the cache.
"""
import os
import sys

from npu_lora import backends as B
from npu_lora import cli

K, N = int(os.environ["SHARD"]), int(os.environ["NSHARD"])
orig = B.FuriosaBackend.compile
seen = [0]


def compile_shard(self, key, gm, sample):
    i = seen[0]
    seen[0] += 1
    cached = os.path.exists(self._path(key) + ".edf")
    if i % N != K and not cached:
        print(f"[shard {K}/{N}] skip {key}", flush=True)
        return lambda *a: (_ for _ in ()).throw(RuntimeError("stub part"))
    return orig(self, key, gm, sample)


B.FuriosaBackend.compile = compile_shard
cli.main(sys.argv[1:])
