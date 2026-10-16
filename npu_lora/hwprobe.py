"""장치 전송 대역폭과 '계산 중 전송' 겹치기 가능 여부를 잰다 — offload 학습 설계의 입력값.

  python -m npu_lora.hwprobe --backend furiosa:auto
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="furiosa:auto")
    ap.add_argument("--mb", type=int, default=117, help="텐서 하나 크기(MB, bf16)")
    a = ap.parse_args()
    from . import backends, graph as G
    be = backends.make(a.backend)
    n = a.mb * 1024 * 1024 // 2 // 4096
    x = torch.randn(n, 4096).to(torch.bfloat16)
    k = 8
    t0 = time.time()
    d = [be.to_device(x) for _ in range(k)]
    be.sync()
    up = time.time() - t0
    t0 = time.time()
    h = [be.to_host(t) for t in d]
    down = time.time() - t0
    gb = k * x.numel() * 2 / 1e9
    a_ = torch.randn(128, 4096).to(torch.bfloat16)
    m = be.compile(f"hwprobe-mm-{n}", G.build(lambda u, w: u @ w.t(), [a_, x]), [a_, x])
    da = be.to_device(a_)
    for _ in range(3):
        m(da, d[0])
    be.sync()
    reps = 50
    t0 = time.time()
    for _ in range(reps):
        m(da, d[0])
    be.sync()
    comp = (time.time() - t0) / reps

    def pull():
        return [be.to_host(t) for t in d[:4]]
    t0 = time.time()
    pull()
    alone = time.time() - t0
    with ThreadPoolExecutor(1) as ex:
        t0 = time.time()
        f = ex.submit(pull)
        for _ in range(reps):
            m(da, d[0])
        be.sync()
        r = f.result()
        both = time.time() - t0
    ok = all(torch.equal(p, q) for p, q in zip(r, h[:4]))
    serial = alone + reps * comp
    print(json.dumps({"h2d_GBps": round(gb / up, 2), "d2h_GBps": round(gb / down, 2), "mm_ms": round(comp * 1e3, 3),
                      "d2h_alone_s": round(alone, 3), "mm_plus_d2h_thread_s": round(both, 3),
                      "serial_s": round(serial, 3), "overlap_ratio": round((serial - both) / min(alone, reps * comp), 2),
                      "data_ok": ok, **be.describe()}))


if __name__ == "__main__":
    main()
