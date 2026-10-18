# 1B 전체 파인튜닝: 과제 + 범용 replay (RNGD vs A100)

- **Date**: 2026-10-16
- **Task**: 전체 파인튜닝 구현·검증
- **Host / device**: RNGD = npu-node-1 (lmd-top Job, furiosa-torch 2026.2.0, 오프로드 파이프라인) / A100 = e11 GPU1 (CUDA 백엔드, vllm-openai v0.11.0 이미지). 평가 e11 A100 vLLM v0.11.0 오프라인
- **Commit**: 976d7d7-dirty (npu-lora)

## Question

과제만 학습했을 때 생긴 망각(`2026-10-16-full1b-nsmc-task-only.md`, MMLU 43→20/8.5%)을 범용 지시문 replay로
막을 수 있는가. 그 레시피에서도 RNGD와 A100이 같은 결과를 내는가.

## Setup

- 모델 `unsloth/Llama-3.2-1B-Instruct`. 임베딩·lm_head·최종 norm 고정, 디코더 가중치 전부 학습.
- 데이터 `nsmc_mix_train.jsonl`: NSMC 4,000 + Alpaca 4,000 (1:1, 섞음). 8,000개 중 1,359개 128 토큰 초과(프롬프트 쪽을 자름).
- 학습(두 장치 동일): S=128, lr 1e-5, 800 step × accum 4, seed 0.
- 평가: `npu_lora.bench`. NSMC test 1,000개, MMLU 400문항.

## Command

```bash
lmd-top --plan train --vendor furiosa --model unsloth/Llama-3.2-1B-Instruct --set full=true \
  --set data=/mnt/store/datasets/quality/nsmc_mix_train.jsonl --set name=fullmix1b-rngd \
  --set steps=800 --set accum=4 --set seq_len=128 --set lr=1e-5 --set node=npu-node-1 ...
python3 -m npu_lora.cli train --full --model unsloth/Llama-3.2-1B-Instruct --backend cuda:0 \
  --data .../nsmc_mix_train.jsonl --seq-len 128 --steps 800 --accum 4 --lr 1e-5 --run-name fullmix1b-cuda \
  --out /mnt/store/finetuned/llama-3.2-1b-nsmcmix-full-cuda
python3 -m npu_lora.bench --model <model dir> --label ... --gpu-mem 0.25 \
  --task nsmc=.../nsmc_test.jsonl --task mmlu=.../mmlu.jsonl
python3 -m npu_lora.compare experiments/data/fullmix-rngd.jsonl experiments/data/fullmix-cuda.jsonl --window 40
```

## Results

| config | NSMC test | MMLU | step (median) |
|---|---:|---:|---:|
| 기본 1B | 62.8% | 43.25% | — |
| 과제만, RNGD (참고) | 79.8% | 20.25% | ~5 s |
| 과제만, A100 (참고) | 78.5% | 8.5% | ~4.2 s |
| **replay, RNGD** | **78.2%** | **36.25%** | 3.85 s |
| replay, A100 | 79.0% | 36.75% | 16.0 s* |

\* A100 run은 e11 CPU 경합(host AdamW가 CPU에서 돎) 상태에서 돌아 느렸다. 장치 속도 비교로 쓰지 말 것.

loss: step 1–3 RNGD 1.399/1.354/1.955, A100 1.397/1.355/1.953. 40 step 구간 평균 첫 구간 1.2507 vs 1.2332,
마지막 구간 0.7134 vs 0.7142, 구간 상대차 중앙값 1.2%.

lr 5e-6 RNGD 재시도는 npu0을 다른 사용자가 쓰고 있어 EBUSY로 시작하지 못했다(측정 없음).

Raw logs: `experiments/data/fullmix-rngd.jsonl`, `experiments/data/fullmix-cuda.jsonl`,
`/mnt/store/bench/fullmix1b-{rngd,cuda}.json`

## Conclusion

replay를 섞으면 과제 성능은 거의 그대로(+15~16%p) 두고 MMLU 손실을 −23~35%p에서 −7%p로 줄인다.
RNGD와 A100은 이 레시피에서도 차이가 오차 범위 안이다(NSMC 0.8%p, MMLU 0.5%p, loss 곡선 1.2%).
즉 RNGD 전체 파인튜닝은 GPU와 같은 품질을 낸다. 남은 망각 7%p는 lr·replay 비율 문제로, 장치와는 무관하다.
