"""품질 검증용 데이터: NSMC(네이버 영화 리뷰 감성, 공개 train/test 분리) + MMLU 일부(망각 확인).

  python examples/nsmc.py OUT_DIR [--train 4000] [--test 1000] [--mmlu 400]

OUT_DIR/nsmc_train.jsonl  {"prompt","response"}  — 학습
OUT_DIR/nsmc_test.jsonl   {"prompt","expect"}    — 학습에 없는 리뷰로 정답률
OUT_DIR/mmlu.jsonl        {"prompt","expect"}    — 학습 전후 일반 지식 비교(정답 글자)
OUT_DIR/nsmc_mix_train.jsonl — NSMC 와 범용 지시문(Alpaca)을 1:1 로 섞은 학습 데이터(--replay N).
    전체 파인튜닝을 한 과제로만 하면 모든 질문에 그 과제 형식('부정')으로 답하는 망각이 생긴다(1B: MMLU
    43% → 8~20%). 범용 지시문을 섞는 replay 가 표준 대응이다.
"""
import argparse
import csv
import io
import json
import os
import random
import urllib.request

NSMC = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt"
MMLU = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
ALPACA = "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
INSTR = "다음 영화 리뷰의 감정을 '긍정' 또는 '부정' 한 단어로만 답하세요.\n리뷰: {}"
LABEL = {"1": "긍정", "0": "부정"}


def nsmc(split):
    raw = urllib.request.urlopen(NSMC.format(split), timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
    return [r for r in rows if r["document"] and len(r["document"]) >= 5]


def balanced(rows, n, rng):
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[: n // 2] + neg[: n // 2]
    rng.shuffle(out)
    return out


def mmlu(n, rng):
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(MMLU, timeout=120).read()))
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        r = df.iloc[i]
        choices = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
        q = (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n"
             f"{r['question']}\n{choices}\nAnswer with only the letter (A, B, C, or D).")
        out.append({"prompt": q, "expect": "ABCD"[int(r["answer"])], "subject": r["subject"]})
    return out


def alpaca(n, rng):
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(ALPACA, timeout=120).read()))
    out = []
    for i in rng.sample(range(len(df)), n):
        r = df.iloc[i]
        prompt = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
        out.append({"prompt": prompt, "response": r["output"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--mmlu", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", type=int, default=4000, help="섞을 Alpaca 예제 수(0=만들지 않음)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tr, te = balanced(nsmc("train"), a.train, rng), balanced(nsmc("test"), a.test, rng)
    with open(os.path.join(a.out, "nsmc_train.jsonl"), "w") as f:
        for r in tr:
            f.write(json.dumps({"prompt": INSTR.format(r["document"]), "response": LABEL[r["label"]]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "nsmc_test.jsonl"), "w") as f:
        for r in te:
            f.write(json.dumps({"prompt": INSTR.format(r["document"]), "expect": LABEL[r["label"]]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "mmlu.jsonl"), "w") as f:
        for r in mmlu(a.mmlu, rng):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.replay:
        nsmc_rows = [{"prompt": INSTR.format(r["document"]), "response": LABEL[r["label"]]} for r in tr]
        mix = nsmc_rows + alpaca(a.replay, rng)
        rng.shuffle(mix)
        with open(os.path.join(a.out, "nsmc_mix_train.jsonl"), "w") as f:
            for r in mix:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"nsmc train {len(tr)}, test {len(te)}, mmlu {a.mmlu}, mix {len(tr) + a.replay if a.replay else 0} -> {a.out}")


if __name__ == "__main__":
    main()
