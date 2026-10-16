# 1B 전체 파인튜닝: 과제만 학습 (RNGD vs A100)

- **Date**: 2026-10-16
- **Task**: 전체 파인튜닝 구현·검증
- **Host / device**: RNGD = etri-001 npu0 (lmd-top Job, furiosa-torch 2026.2.0) / A100 = e11 GPU1 (CUDA 백엔드). 평가 e11 A100 vLLM v0.11.0
- **Commit**: f232df7-dirty

## Question

`--full`(레이어 가중치 9종 학습, host fp32 master + AdamW)이 RNGD에서 GPU와 같은 결과를 내는가.

## Setup

- 모델 `unsloth/Llama-3.2-1B-Instruct`(meta 원본은 gated). 임베딩·lm_head(tied)·최종 norm 고정.
- RNGD 기울기 점검: `npu-lora check --full --layers 2 --seq-len 128` → loss rel 3.8e-5, 가중치 기울기 18개 최대 rel 2.1%, cos ≥ 0.9998 (PASS).
- 학습(두 장치 동일): NSMC train만, S=128, lr 1e-5, 400 step × accum 4, seed 0.

## Command

```bash
lmd-top --plan train --vendor furiosa --model unsloth/Llama-3.2-1B-Instruct --set full=true \
  --set data=.../nsmc_train.jsonl --set name=full1b-rngd --set steps=400 --set accum=4 --set seq_len=128 --set lr=1e-5 ...
python3 -m npu_lora.cli train --full --model unsloth/Llama-3.2-1B-Instruct --backend cuda:0 ... (같은 인자)
python3 -m npu_lora.bench --model <full model dir> --label ... --task nsmc=... --task mmlu=...
```

## Results

| config | NSMC test | MMLU | step |
|---|---:|---:|---:|
| 기본 1B | 62.8% | 43.25% | — |
| RNGD full FT | 79.8% | 20.25% | ~5 s (동기 갱신, etri-001 64코어) |
| A100 full FT | 78.5% | 8.5% | ~4.2 s |

처음 3 step loss: RNGD 0.511/0.362/2.401, A100 0.543/0.360/2.379. 40 step 구간 상대차 중앙값 6.7%.
MMLU 답 샘플: 둘 다 '부정' 등 NSMC 형식으로 답함(과적합 망각).

Raw logs: `experiments/data/full-rngd.jsonl`, `experiments/data/full-cuda.jsonl`, `/mnt/store/bench/full1b-{rngd,cuda}.json`

## Conclusion

NPU 전체 파인튜닝 자체는 GPU와 같은 방향으로 동작한다(과제 +16~17%p). 그러나 과제 하나만으로 전체 가중치를
학습하면 양쪽 모두 일반 능력이 무너진다 — NPU 문제가 아니라 레시피 문제. 대응으로 범용 지시문(Alpaca) 1:1
replay 데이터를 만들어 재실험(`2026-10-16-full1b-nsmc-replay.md`).
