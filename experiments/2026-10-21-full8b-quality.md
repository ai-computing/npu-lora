# Llama-3.1-8B 전체 파인튜닝 품질: RNGD 학습 vs A100 학습 (replay 레시피)

- **Date**: 2026-10-20 ~ 2026-10-21
- **Task**: #23
- **Host / device**: 학습 npu-node-2 — RNGD 1장(npu0, `furiosaai/furiosa-llm:latest` digest `sha256:37a85288…`, furiosa-torch 2026.2.0,
  드라이버·펌웨어 2026.4.0) / A100 80GB PCIe 1장(`vllm/vllm-openai:v0.11.0`, torch 2.8.0+cu128). 평가 npu-node-2 A100, vLLM 0.11.0 오프라인.
- **Commit**: RNGD `b3a5e90`, A100 `16e17aa` (perf/21-resident-host-buffers, #26; 차이는 장치 메모리 해제 시점뿐).
  학습 실행은 #22 에너지 측정과 같은 실행이다.

## Question

8B 전체 파인튜닝을 RNGD 1장 + CPU 오프로드로 학습하면 같은 코드·조건의 A100 학습과 같은 품질이 나오는가.
1B 에서 쓴 replay 레시피(NSMC + Alpaca 1:1)가 8B 에서도 일반 능력(MMLU)을 지키는가. LoRA 와 비교하면 어떤가.

## Setup

- 모델 `meta-llama/Llama-3.1-8B-Instruct`, 디코더 가중치 전부 학습(임베딩·lm_head·최종 norm 고정), fp32 master·AdamW 는 host.
- 데이터 `nsmc_mix_train.jsonl`(NSMC 4,000 + Alpaca 4,000, `examples/nsmc.py`), S=512 패킹, accum 4, 200 step(409,600 토큰 —
  1B replay 실행의 800 step × 4 × 128 과 같은 토큰 수), lr 1e-5(warmup 10, cosine), grad clip 1.0, seed 0. 두 장치 동일.
- 평가 `npu_lora.bench`: chat 템플릿, temperature 0, 8 토큰. NSMC 시험 1,000개, MMLU 400문항. 기본 모델과 LoRA 어댑터
  (2026-10-16 의 400 step RNGD/A100 어댑터)를 같은 날 같은 vLLM 으로 다시 평가했다.

## Command

```bash
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --backend furiosa:auto --cache-dir /cache/furiosa --seq-len 512 --steps 200 --accum 4 --lr 1e-5 --seed 0 \
  --run-name w7-full8b-rngd --out /mnt/store/w7/out/full8b-rngd            # A100: --backend cuda:0, --out .../full8b-cuda
T="--task nsmc=/mnt/store/datasets/quality/nsmc_test.jsonl --task mmlu=/mnt/store/datasets/quality/mmlu.jsonl"
python3 -m npu_lora.bench --model /mnt/store/w7/out/full8b-rngd --label full8b-rngd $T --out full8b-rngd.json
python3 -m npu_lora.bench --model /mnt/store/w7/out/full8b-cuda --label full8b-cuda $T --out full8b-cuda.json
python3 -m npu_lora.bench --model meta-llama/Llama-3.1-8B-Instruct --adapter lora400-rngd=.../nsmc-rngd \
  --adapter lora400-cuda=.../nsmc-cuda --adapter lora200-cuda-w7=/mnt/store/w7/out/lora8b-cuda $T --out base8b-lora.json
python3 -m npu_lora.compare experiments/data/w7/energy-full8b-rngd.jsonl experiments/data/w7/energy-full8b-a100.jsonl --window 40
```

## Results

| Llama-3.1-8B-Instruct | NSMC 시험 (n=1000) | MMLU (n=400) | 학습 tok/s |
|---|---:|---:|---:|
| 기본 모델 | 79.6% | 60.75% | — |
| LoRA r=16 q·v, 400 step, RNGD 학습 | 87.2% | 63.0% | — |
| LoRA r=16 q·v, 400 step, A100 학습 | 87.6% | 63.25% | — |
| **전체 FT, replay, 200 step, RNGD 학습** | **84.6%** | **52.75%** | 332 |
| 전체 FT, replay, 200 step, A100 학습 | 84.2% | 50.25% | 492 |

- **RNGD 와 A100 전체 FT 는 같은 품질이다**: NSMC 0.4%p, MMLU 2.5%p 차(n=400 에서 표준오차 ~2.5%p). loss 는 40 step 구간 상대차 중앙값
  0.15 %, 첫 step 1.7157 대 1.7237, 마지막 step 1.0047 대 1.0017, 최저 0.529 대 0.520.
- LoRA 어댑터 재평가는 10-16 기록(87.5/63.0, 87.6/63.0)과 0.3%p 이내로 같다(vLLM 배치 비결정성).
- **8B 에서는 replay 가 망각을 다 막지 못한다**: MMLU −8 ~ −10.5%p(1B 는 −7%p). NSMC 는 +5%p 로 LoRA(+7.6~8%p)보다 작다.
  같은 토큰 수에서 전체 FT 는 LoRA 보다 과제 성능이 낮고 일반 능력 손실이 크다.

Raw: `experiments/data/w7/energy-full8b-{rngd,a100}.jsonl`(학습 로그), `experiments/data/w7/bench-*.json`(평가, 문항 예시 포함).

## Conclusion

8B 전체 파인튜닝에서도 RNGD 학습은 A100 학습과 같은 결과를 낸다(loss 곡선 0.15 %, 평가 차이 오차 범위). 즉 논문 C2(정확한 기울기로
GPU 와 같은 품질)가 LoRA 에서 전체 FT 까지 확장된다. 다만 이 레시피(lr 1e-5, 200 step, Alpaca 1:1)는 8B 에서 MMLU 를 8~10%p 잃어
LoRA 보다 나쁘다 — 장치 문제가 아니라 레시피 문제다. 다음: lr 5e-6 또는 replay 비율 2:1, 그리고 시드 반복.
