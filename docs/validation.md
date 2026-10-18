# 검증 결과

실험별 원자료와 명령은 `experiments/` 에 있다.

## PoC (2026-10-15, ETRI llm-d 클러스터)

| 항목 | 결과 |
| --- | --- |
| 함수형 Llama vs HF `LlamaForCausalLM` | loss 상대오차 1.7e-5 (q·k·v·o LoRA, 합친 모델 경유) |
| RNGD vs CPU 기울기 (Llama-3.1-8B 2 레이어 + 128k 어휘 head) | loss 1.9e-4, 기울기 cos ≥ 0.9997, 상대오차 ≤ 2.5% |
| RNGD 1장 전체 학습 (Llama-3.1-8B 32 레이어, r=16 q·v, S=128) | 0.7 s/step(약 180 tok/s), 200 step 2분 36초, loss 3.29 → ~0 |
| 학습 전 → 후 평가 (학습에 없는 질문 10개, A100 vLLM 런타임 LoRA) | 0/10 → **10/10** (CUDA 백엔드 학습도 10/10) |
| 첫 컴파일 (레이어 순전파·역전파 + head, 조각 단위) | 약 1.5 시간, 이후 EDF 캐시(15 MB)로 즉시 시작 |

## 품질 검증 — RNGD 학습이 GPU 학습과 같은 품질인가 (2026-10-16)

같은 데이터(NSMC 학습 4,000개 중 1,600개 분량)·시드·설정(S=128, r=16 q·v, lr 2e-4, 400 step × accum 4)으로
RNGD 와 A100 에서 각각 LoRA 학습 → vLLM 오프라인 평가(`npu_lora.bench`).

| | NSMC 시험 정답률 (n=1000, 학습에 없는 리뷰) | MMLU (n=400, 망각 확인) |
| --- | --- | --- |
| 기본 Llama-3.1-8B-Instruct | 79.6% | 60.8% |
| **RNGD 학습 LoRA** | **87.5%** | **63.0%** |
| A100 학습 LoRA (같은 조건) | 87.6% | 63.0% |

loss 곡선(40 step 구간 평균)의 RNGD/A100 상대차 중앙값 2.8%, 마지막 구간 0.0676 vs 0.0667
(`python -m npu_lora.compare a/train.jsonl b/train.jsonl`). 이 검증 과정에서 고친 품질 버그:
Llama 3.x 의 llama3 `rope_scaling` 미적용(HF 대비 1.8e-4 → 3.5e-7), 긴 예제가 응답째 잘리던 문제(이제 프롬프트 쪽을 덜어냄).

재현: `examples/nsmc.py OUT` → `lmd-top --plan train … --set data=…/nsmc_train.jsonl` →
`python -m npu_lora.bench --model … --adapter rngd=… --task nsmc=… --task mmlu=…`.

데모: `examples/cluster_facts.py` 로 만든 데이터를 `lmd-top --plan train … --apply` 로 학습하고
`npu-lora serve` → `npu-lora eval`.

## 전체 파인튜닝 (Llama-3.2-1B, 2026-10-16)

| | NSMC test | MMLU | 비고 |
| --- | ---: | ---: | --- |
| 기본 1B | 62.8% | 43.25% | |
| RNGD full FT, NSMC 만 | 79.8% | 20.25% | 과제 하나로 전체 학습 → 모든 질문에 과제 형식으로 답하는 망각 |
| A100 full FT, NSMC 만 | 78.5% | 8.5% | 같은 망각(장치 문제가 아님) |
| RNGD full FT, NSMC + Alpaca 1:1 | 78.2% | 36.25% | replay 로 망각이 크게 줄었다 |
| A100 full FT, NSMC + Alpaca 1:1 | 79.0% | 36.75% | RNGD 와 오차 범위 안 (loss 곡선 상대차 1.2%) |

RNGD 전체 파인튜닝 기울기 점검(`npu-lora check --full`, 2층): loss 상대오차 3.8e-5, 가중치 기울기 최대 2.1%.
자세한 조건은 `experiments/2026-10-16-full1b-nsmc-task-only.md`.
