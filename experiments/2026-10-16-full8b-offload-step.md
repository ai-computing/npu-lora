# 8B 전체 파인튜닝: RNGD 1장 + CPU 오프로드 파이프라인 step 시간

- **Date**: 2026-10-16
- **Task**: NPUTRAIN-19 8B full FT A100 대비 속도·품질 비교
- **Host / device**: npu-node-2 RNGD 1장(furiosa:auto, 공식 furiosa-llm 이미지, furiosa-torch 2026.2.0), pod CPU request 16, host load avg 37~72 (같은 노드에서 EAGLE 학습 진행 중). 컴파일은 npu-node-1(compile-only)에서 7,060 s
- **Commit**: 0534900-dirty (npu-lora)

## Question

카드 1장(48 GB)에서 Llama-3.1-8B 전체 파인튜닝(디코더 가중치 7.0B개)이 도는가, step 시간은 얼마이고 병목은 무엇인가.

## Setup

- 장치: bf16 가중치 + 누적 기울기만. host: fp32 master + AdamW 상태(fused CPU AdamW).
- 레이어별 작업(ThreadPool 4): 기울기 D2H → 전역 clip 대기 → AdamW → bf16 H2D. 다음 step forward는 레이어마다 자기 갱신만 기다린다.
- 데이터 nsmc_mix_train, S=512, lr 1e-5. accum 1은 30 step(6~30 step 중앙값), accum 4는 20 step(4~20 step 중앙값).
  accum 4는 기울기를 장치에서 누적하는 별도 그래프(full-bwd-acc, 39조각)를 npu-node-1에서 6,617 s 동안 컴파일했다.

## Command

```bash
# npu-node-1 (장치 없이 컴파일)
python -m npu_lora.cli compile --full --model meta-llama/Llama-3.1-8B-Instruct --seq-len 512 --accum 1
# e11 RNGD pod
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct \
  --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl --backend furiosa:auto \
  --seq-len 512 --steps 30 --accum 1 --lr 1e-5 --run-name pipe8b --out /tmp/out8b
# accum 4: compile/train 모두 --accum 4, --steps 20 --run-name pipe8b4
```

## Results

| 항목 | 값 |
|---|---:|
| step 시간 (중앙값) | **10.33 s** (49.6 tok/s) |
| 그중 장치 계산(forward+backward+head) | ~1.5 s (step 1, 대기 없음: 1.60 s) |
| 그중 forward가 직전 갱신을 기다린 시간 | 8.78 s |
| 레이어 작업 합(4 스레드 합산): D2H / AdamW / H2D | 17.97 / 7.62 / 9.22 s |
| → 벽시계 환산(÷4) | 4.5 / 1.9 / 2.3 s ≈ 8.7 s |

- 한 step에 오가는 양: 기울기 14 GB 내림 + 가중치 14 GB 올림. 환산 대역 D2H ≈ 3.1 GB/s, H2D ≈ 6.1 GB/s로
  hwprobe 측정 상한(3.1 / 6.4 GB/s)과 같다. **병목은 PCIe 전송**이고 장치 계산은 15% 정도다.
- loss 1.80 → 30 step 최저 0.28, 발산 없음. 결과 모델 저장까지 정상 종료.

### accum 비교

| accum | 토큰/step | step 시간 | 처리량 | 가중치 대기 |
|---:|---:|---:|---:|---:|
| 1 | 512 | 10.33 s | 49.6 tok/s | 8.78 s |
| 4 | 2,048 | 13.13 s | **156.0 tok/s** | 8.61 s |

accum 4에서 장치 계산은 micro-batch당 ~1.1 s(13.13 − 8.61 = 4.5 s / 4)이고 갱신 전송 고정비(~8.6 s)는 그대로다.
처리량이 3.1배가 됐다. 같은 식이면 accum 8은 ~17.6 s / 4,096 토큰 ≈ 230 tok/s다(추정, 미측정).
loss(accum 4, 20 step): 1.72 → 0.82~1.56, 발산 없음.

Raw logs: `experiments/data/pipe8b-rngd.jsonl`, `experiments/data/pipe8b4-rngd.jsonl`

## Conclusion

8B 전체 파인튜닝이 RNGD 1장에서 돈다(카드에 16 GB 가중치 + 16 GB 기울기, host 약 100 GB).
step 시간은 갱신 전송이 정한다(~8.7 s/step 고정비). 따라서 빨라지는 길은
(1) gradient accumulation으로 고정비를 여러 micro-batch에 나누기(accum 4에서 156 tok/s, 3.1배 확인),
(2) 카드 2장에 레이어를 나눠 PCIe 링크를 두 개 쓰기(NPUTRAIN-18)다.
A100 대조는 e11 GPU 두 장을 다른 학습(EAGLE)이 쓰고 있어 아직 못 했다.
