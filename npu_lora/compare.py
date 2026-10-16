"""두 학습 로그(JSONL)의 loss 곡선 비교 — 같은 데이터·시드·설정이면 장치가 달라도 곡선이 겹쳐야 한다.

  python -m npu_lora.compare a/train.jsonl b/train.jsonl [--window 20]

출력: step 구간별 평균 loss(두 곡선), 구간 평균의 상대차, 전체 지표.
bf16 연산 순서가 장치마다 달라 step 단위로는 갈라지지만(혼돈적 누적), 구간 평균은 오차 범위 안이어야 한다.
"""
import argparse
import json


def load(path):
    rows = [json.loads(l) for l in open(path) if '"event": "step"' in l]
    return {r["step"]: r["loss"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    la, lb = load(a.a), load(a.b)
    steps = sorted(set(la) & set(lb))
    rows, diffs = [], []
    for s0 in range(steps[0], steps[-1] + 1, a.window):
        w = [s for s in steps if s0 <= s < s0 + a.window]
        ma = sum(la[s] for s in w) / len(w)
        mb = sum(lb[s] for s in w) / len(w)
        d = abs(ma - mb) / max(abs(mb), 1e-6)
        diffs.append(d)
        rows.append({"steps": f"{w[0]}-{w[-1]}", "a": round(ma, 4), "b": round(mb, 4), "rel": round(d, 3)})
    first = [s for s in steps if s <= a.window]
    last = [s for s in steps if s > steps[-1] - a.window]
    out = {"steps": len(steps), "windows": rows,
           "first_window": [round(sum(la[s] for s in first) / len(first), 4), round(sum(lb[s] for s in first) / len(first), 4)],
           "last_window": [round(sum(la[s] for s in last) / len(last), 4), round(sum(lb[s] for s in last) / len(last), 4)],
           "median_window_rel": sorted(diffs)[len(diffs) // 2]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
