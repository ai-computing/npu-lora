# NSMC LoRA 품질: RNGD 학습 vs A100 학습 (Llama-3.1-8B)

- **Date**: 2026-10-16
- **Task**: 파인튜닝 품질 검증 (RNGD vs A100)
- **Host / device**: 학습 RNGD = npu-node-1 npu0 (furiosa-torch 2026.2.0, 공식 furiosa-llm 이미지, lmd-top Job) / A100 = npu-node-2 GPU1 (vllm-openai v0.11.0 이미지, npu-lora CUDA 백엔드). 평가 = e11 A100, vLLM v0.11.0 오프라인
- **Commit**: f232df7-dirty (npu-lora 미커밋)

## Question

RNGD에서 학습한 LoRA가 같은 조건의 GPU 학습과 같은 품질(일반화·망각)을 내는가.

## Setup

- 데이터: `examples/nsmc.py` — NSMC 공개 train에서 균형 4,000개, test에서 1,000개, MMLU test 400문항(망각 확인). 시드 0.
- 학습(두 장치 동일): Llama-3.1-8B-Instruct, S=128, LoRA r=16 α=32 q·v, lr 2e-4(warmup 10, cosine), 400 step × accum 4, grad clip 1.0, seed 0. 4,000개 중 291개는 128 토큰 초과로 잘림(당시 뒤를 자름 → 이후 수정).
- 평가: `npu_lora.bench` — chat 템플릿, temperature 0, max 8 토큰. NSMC는 첫 '긍정/부정', MMLU는 첫 A–D.
- 이 실험 직전에 llama3 `rope_scaling` 미적용 버그를 고쳤다(HF 대비 1.8e-4 → 3.5e-7, 실제 8B 2층 S=512).

## Command

```bash
# RNGD
lmd-top --plan train --vendor furiosa --model meta-llama/Llama-3.1-8B-Instruct \
  --set data=/mnt/store/datasets/quality/nsmc_train.jsonl --set name=nsmc-rngd --set steps=400 --set accum=4 \
  --set seq_len=128 --set lr=2e-4 --set node=npu-node-1 --set hf_host_path=/home/user/.cache/huggingface \
  --set cache_host_path=/home/user/.cache/npu-lora --mode admin --apply
# A100 (pod on e11 GPU1)
python3 -m npu_lora.cli train --model meta-llama/Llama-3.1-8B-Instruct --data .../nsmc_train.jsonl --backend cuda:0 \
  --seq-len 128 --steps 400 --accum 4 --lr 2e-4 --rank 16 --alpha 32 --targets q,v --run-name nsmc-cuda --out ...
# 평가
python3 -m npu_lora.bench --model meta-llama/Llama-3.1-8B-Instruct --adapter rngd=.../nsmc-rngd --adapter gpu=.../nsmc-cuda \
  --task nsmc=.../nsmc_test.jsonl --task mmlu=.../mmlu.jsonl --out /mnt/store/bench/nsmc-lora.json
python3 -m npu_lora.compare experiments/data/nsmc-rngd.jsonl experiments/data/nsmc-cuda.jsonl --window 40
```

## Results

| config | NSMC test (n=1000) | MMLU (n=400) | step 시간 |
|---|---:|---:|---:|
| 기본 모델 | 79.6% | 60.75% | — |
| **RNGD LoRA** | **87.5%** | **63.0%** | 2.6 s (512 tok) |
| A100 LoRA | 87.6% | 63.0% | ~2.8 s |

loss 곡선(40 step 구간 평균): 첫 구간 0.1620 vs 0.1621, 마지막 구간 0.0676 vs 0.0667, 구간 상대차 중앙값 2.8%.
step 1 loss 0.6956(RNGD) vs 0.7012(A100).

Raw logs: `experiments/data/nsmc-rngd.jsonl`, `experiments/data/nsmc-cuda.jsonl`, 평가 `/mnt/store/bench/nsmc-lora.json`

## Conclusion

RNGD 학습 LoRA는 A100 학습과 오차 범위 안에서 같은 품질이다(NSMC 0.1%p 차, 표준오차 ~1%p). 학습에 없는
리뷰로 +7.9%p 일반화하고 MMLU는 떨어지지 않는다(+2.2%p, 짧은 답 형식 준수 효과로 보임). 남은 것: 같은 비교를
시드 여러 개로 반복해 분산 확인, 더 긴 학습(1 epoch 이상).
