# 같은 조건의 학습 에너지·속도: RNGD 1장 vs A100 1장 (Llama-3.1-8B LoRA, full FT)

- **Date**: 2026-10-20 ~ 2026-10-21
- **Task**: #22 (W7 논문 §5.4·§7 "같은 조건의 tok/J")
- **Host / device**: npu-node-2 한 노드에서 RNGD 1장(npu0)과 A100 80GB PCIe 1장(GPU0)을 번갈아 썼다(노드 CPU 24코어, 메모리 1 TB).
  RNGD: 공식 `furiosaai/furiosa-llm:latest`(digest `sha256:37a85288…`, furiosa-torch 2026.2.0, torch 2.10.0+cpu, transformers 5.1.0),
  드라이버·펌웨어 2026.4.0. A100: `vllm/vllm-openai:v0.11.0`(torch 2.8.0+cu128, transformers 4.57.0), NVIDIA 드라이버 580.178.04.
  모든 실행은 Kubernetes Job(장치 플러그인이 카드 배정)이고, 측정 동안 그 노드에는 이 실험의 pod 하나만 돌았다.
- **Commit**: perf/21-resident-host-buffers (#26). A100 실행 3건은 `16e17aa`(상주 host 버퍼), RNGD 실행은 `b3a5e90`
  (+ 이전 장치 가중치 해제 — 없으면 RNGD 8B full FT 가 16 step 에서 장치 메모리 부족으로 죽었다,
  `experiments/data/w7/full8b-rngd-oom-before-fix.txt`). 두 커밋의 차이는 장치 메모리 해제 시점뿐이다.
  A100 PyTorch 기준선은 `experiments/data/w7/native_lora_a100.py`.

## Question

같은 레시피(데이터·시드·하이퍼파라미터)로 RNGD 1장과 A100 1장의 학습 처리량(tok/s), 카드 평균 전력(W), tok/J 는 얼마인가.
CPU 오프로드 경로(full FT)에서는 host 쪽 전력을 넣으면 비교가 어떻게 바뀌는가.

## Setup

- **LoRA**: Llama-3.1-8B-Instruct, README 레시피 — S=128, r=16 α=32 q·v, lr 2e-4(warmup 10, cosine), accum 4, grad clip 1.0, seed 0,
  NSMC train(`examples/nsmc.py`, 4,000개). step 당 512 토큰(패딩 포함, npu-lora 의 tok/s 정의).
  - RNGD: npu-lora `furiosa:auto`, 200 step 계획 중 60 step 에서 끊었다(아래 "RNGD LoRA 속도" 참고 — 정상 상태 구간 55 step, 571 s).
  - A100 (npu-lora CUDA 백엔드): 같은 코드·그래프 구조, 200 step.
  - A100 (PyTorch 기준선): HF 모델 bf16 + q·v LoRA 모듈(PEFT 초기화) + autograd + fused CUDA AdamW, 같은 데이터 패킹(블록 대각 마스크)·스케줄·clip.
    전력 표본을 충분히 얻으려고 1,200 step 돌렸다(첫 step loss 0.7071, npu-lora A100 0.7012, RNGD 0.7101 — 같은 데이터 순서).
- **Full FT**: Llama-3.1-8B-Instruct, S=512, accum 4(step 당 2,048 토큰), lr 1e-5, seed 0, `nsmc_mix_train.jsonl`(NSMC 4k + Alpaca 4k, replay 레시피),
  200 step, 상주 host 버퍼 오프로드 경로(fp32 master·AdamW 는 host). 같은 실행이 #23 품질 평가의 학습이다.
- **전력**: 클러스터 Prometheus 원시 표본(15 s 간격)을 학습 로그 타임스탬프의 정상 구간(6번째 step 끝 ~ 마지막 step)에서 평균했다.
  - RNGD `furiosa_npu_hw_power{label="rms",device="npu0"}`, A100 `DCGM_FI_DEV_POWER_USAGE{gpu="0"}`,
    노드 입력 전력 `ipmi_dcmi_power_consumption_watts`(BMC DCMI, PSU 입력 = 노드 전체).
  - 유휴 기준: 각 날짜에 실행 전 5분(같은 계열). A100 실행 날 노드 605.3 W·GPU0 58.4 W, RNGD 실행 날 노드 624.8 W·npu0 39.0 W
    (이날은 쓰지 않은 GPU1 이 유휴인데도 ~108 W 를 계속 써 노드 유휴가 높다).
  - **host 몫** = (노드 전력 − 노드 유휴) − Σ(노드의 가속기 4장 각각의 전력 − 유휴). **카드+host** = 카드 전력 + host 몫.
    (쓰지 않는 A100 의 유휴 전력이 50 W 와 108 W 사이를 오가서 — RNGD full FT 동안 GPU0 이 49.8 → 100.3 W — 가속기 4장 모두를 뺀다.)
    PSU 효율·팬을 포함한 노드 입력 기준이라 근사다. node_hwmon/RAPL 의 CPU 패키지 전력은 이 노드에서 수집되지 않는다.

## Command

```bash
# Job 매니페스트: experiments/data/w7/jobgen.py (npu_lora.k8s 기반, 실험 노드 taint 용 toleration 추가)
python3 -m npu_lora.cli train --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_train.jsonl \
  --backend furiosa:auto --cache-dir /cache/furiosa --seq-len 128 --steps 200 --accum 4 --lr 2e-4 --rank 16 --alpha 32 \
  --targets q,v --seed 0 --run-name w7-lora8b-rngd --out /mnt/store/w7/out/lora8b-rngd     # A100: --backend cuda:0
python3 experiments/data/w7/native_lora_a100.py --model meta-llama/Llama-3.1-8B-Instruct \
  --data /mnt/store/datasets/quality/nsmc_train.jsonl --seq-len 128 --steps 1200 --accum 4 --lr 2e-4
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --backend furiosa:auto --cache-dir /cache/furiosa --seq-len 512 --steps 200 --accum 4 --lr 1e-5 --seed 0 \
  --run-name w7-full8b-rngd --out /mnt/store/w7/out/full8b-rngd                             # A100: --backend cuda:0
# 전력: Prometheus 포트포워드 후
kubectl -n monitoring port-forward svc/prometheus-operated 19090:9090 &
python3 experiments/data/w7/power.py <train stdout> <node> <out prefix> --skip 5 --idle-at <unix ts> --idle-before 300
```

## Results

### LoRA (S=128, accum 4)

| 장치 · 구현 | tok/s | 카드 W (유휴) | **tok/J (카드)** | 노드 W (유휴) | host 몫 W | tok/J (카드+host) |
|---|---:|---:|---:|---:|---:|---:|
| RNGD · npu-lora | 49.3 | 46.8 (39.0) | **1.05** | 661.7 (624.8) | 30.4 | 0.64 |
| A100 · npu-lora CUDA 백엔드 | 365.9 | 151.8 (58.4) | **2.41** | 732.3 (605.3) | 44.7 | 1.86 |
| A100 · PyTorch (HF + LoRA, autograd) | 1,682.8 | 263.5 (58.4) | **6.39** | 873.7 (605.3) | 74.9 | 4.97 |

### Full FT (S=512, accum 4, CPU 오프로드)

| 장치 | tok/s | 카드 W (유휴) | **tok/J (카드)** | 노드 W (유휴) | host 몫 W | tok/J (카드+host) |
|---|---:|---:|---:|---:|---:|---:|
| **RNGD** | 332.0 | 76.9 (39.0) | **4.32** | 807.3 (624.8) | 96.2 | **1.92** |
| A100 (같은 오프로드 코드) | 492.4 | 160.1 (58.4) | **3.08** | 853.6 (605.3) | 158.3 | 1.55 |

A100 full FT step 은 4.07–4.19 s(그중 다음 순전파가 갱신을 기다린 시간 2.9–3.0 s), 레이어 작업 합(4 스레드) D2H 5.4 s · AdamW 4.5 s · H2D 2.5 s.
RNGD full FT step 은 중앙값 6.36 s(대기 3.04 s), 레이어 작업 합 D2H 4.0 s · AdamW 7.1 s · H2D 3.2 s. 2026-10-19 미커밋 측정(315 tok/s)과 같은 수준이다.
두 장치의 loss 는 일치한다: step 1–3 RNGD 1.7157 / 1.5626 / 1.0830, A100 1.7237 / 1.5647 / 1.0824, 40 step 구간 상대차 중앙값 0.15 %
(`npu_lora.compare`), 마지막 step 1.0047 대 1.0017. 정상 구간은 RNGD 195 step·1,203 s(표본 80개), A100 195 step·811 s(표본 54개).

### RNGD LoRA 속도 (회귀)

이번 RNGD LoRA step 은 10.7–11.4 s(512 토큰)로, 2026-10-16 기록(`2026-10-16-nsmc-lora-rngd-vs-a100.md`, `f232df7`)의 2.6 s 보다 4배 느리다.
값은 맞다: step 1–3 loss 0.7101 / 0.1055 / 2.7018 대 A100 0.7012 / 0.0987 / 2.6911, step 10 의 급등(1.80 대 1.83)도 같다.
그 사이 `main` 의 그래프 분할이 바뀌었다 — 같은 S=128 r16 q·v 레이어가 지금은 순전파 15조각·역전파 39조각이고(컴파일 로그
`experiments/data/w7/compile-*.parts.txt`), import 당시 README 는 8·19조각이었다. 조각마다 장치 호출과 host 왕복이 생기므로
호출 수가 두 배로 늘고 카드가 거의 쉰다(평균 46.8 W, 유휴 39 W). 그래서 **위 RNGD LoRA tok/J 는 지금 `main` 의 값이지 RNGD 의 한계가 아니다.**
10-16 의 2.6 s/step(197 tok/s)이 같은 전력이라면 4.2 tok/J 가 되지만, 그때는 전력을 재지 않았으므로 추정이다.
첫 컴파일: 8B LoRA S=128 6,725 s, 8B full S=512 accum 4 14,690 s, Qwen2.5-1.5B LoRA 6,339 s(조각을 여러 pod 에 나눠 병렬로 채웠다;
나누는 도구는 `experiments/data/w7/shard_compile.py`).

Raw: `experiments/data/w7/energy-*.jsonl`(학습 로그), `energy-*.power.json`(요약), `energy-*.power.csv`(원시 표본).

## Conclusion

- **같은 npu-lora 코드로는** RNGD LoRA 1.05 tok/J, A100 2.41 tok/J 이고, A100 을 GPU 답게(HF+autograd) 돌리면 6.39 tok/J 다.
  지금 `main` 에서는 RNGD 가 에너지 효율에서 A100 을 이기지 못한다. 원인은 장치가 아니라 조각 호출 수로, 카드 전력이 유휴보다 8 W 높을 뿐이다.
  논문의 "저전력" 주장은 이 수치로는 뒷받침되지 않으며, 조각 수 회귀를 고친 뒤 다시 재야 한다.
- **Full FT(오프로드)에서는 RNGD 가 더 효율적이다**: 카드 기준 4.32 대 3.08 tok/J(1.40배), host 를 넣어도 1.92 대 1.55 tok/J(1.24배).
  속도는 A100 이 1.48배 빠르다(492 대 332 tok/s). 두 장치 모두 step 의 절반가량이 host 갱신 대기라 카드 전력이 낮다(RNGD 77 W, A100 160 W).
- **host 전력**: 오프로드 full FT 에서 host 몫이 카드 증가분보다 크다(A100 158 W 대 102 W, RNGD 96 W 대 38 W). host AdamW·전송을 쓰는
  설계에서는 카드 tok/J 만 보면 효율을 과대평가한다. 논문 §5.4 에는 카드+host 열을 함께 실어야 한다.
- 다음: 그래프 분할 회귀 원인 확인(조각 수를 8/19 로 되돌린 빌드로 같은 측정), 같은 조건 반복(시드·날짜)으로 분산 확인.
