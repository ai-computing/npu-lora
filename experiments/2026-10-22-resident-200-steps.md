# 상주 AdamW 200 step (2장 × 상주 8) 과 낮은 load 의 1장 × 상주 4

- **Date**: 2026-10-22 (01:07–01:46 KST)
- **Task**: #40 done-when (200 step, OOM 없음), #34 (1장 상주 4 를 경합 없는 host 에서)
- **Host / device**: npu-node-1 RNGD(카드 3장 중 pod 가 배정받은 1–2장), 노드 CPU 96코어, `furiosaai/furiosa-llm:latest`
  (digest `sha256:37a85288…`, furiosa-torch 2026.2.0, torch 2.10.0+cpu, transformers 5.1.0), 드라이버·펌웨어 2026.4.0. npu-node-2 는 쓰지 않았다.
- **Commit**: main `b3cb03b`(#42 의 DRAM_SHARED 잠금 포함, #43 의 head 변경 포함). pod 에는 이 커밋의 `npu_lora/` 를 ConfigMap 으로 올렸다.
- **Host load**: 1분 load 2장 실행 14.5–50.4, 1장 실행 9.9–43.8(96코어; `data/resident-200/load-npu-node-1.txt`). 10-21 기록들보다 조용하다.

## Question

1. #42 의 잠금으로 상주 실행이 200 step 동안 OOM 없이 도는가(#40 의 done-when)?
2. 경합 없는 host 에서 1장 × 상주 4 의 step 시간과, 같은 시각 1장 오프로드 항으로 계산한 step 모델의 오차는?

## Setup

- Job 하나에 한 측정씩, 차례로(동시에 돌지 않게): (1) 2장 상주 8, 200 step, (2) 1장 오프로드 25 step 뒤 1장 상주 4 25 step(한 Job 에서 이어서).
  Job 마다 `activeDeadlineSeconds`(3600 / 3000)·`ttlSecondsAfterFinished` 를 걸었다.
- 제출 직전마다 확인(`run200.log`): 1분 load < 150, 쿠버네티스에서 필요한 장수가 비어 있음, `/proc/*/fd` 에 `/dev/rngd` 를 연 프로세스 없음.
- 첫 제출은 노드에서 `furiosa-llm:latest` 이미지가 지워져 있어 `ImagePullBackOff` 가 됐고, 카드를 잡은 채 기다리지 않게 Job 을 바로 지웠다.
  카드 요청 없는 pod 로 이미지를 다시 받은 뒤(같은 digest) 다시 확인하고 제출했다.
- #43 이 head 그래프(정답 logit 을 host 에서)를 바꿔서 head 의 ce-stats·ce-grad 조각은 이 실행 pod 안에서 다시 컴파일됐다(캐시 키가 바뀜, 94 s + 398 s).
  그동안 카드 2장을 잡고 있었다 — 다음부터는 main 이 그래프를 바꾸면 카드 없는 CPU Job 으로 `npu-lora compile` 을 먼저 돌린다(1장 Job 은 캐시를 읽어 7–15 s).
  그래서 loss 는 10-21 기록(#43 이전)과 step 1 부터 조금 다르다(1.7157 → 1.7202). 장치 그래프·옵티마이저는 같다.

## Command

```bash
T="python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --cache-dir /cache/furiosa --seq-len 512 --accum 4 --lr 1e-5 --seed 0 --out /tmp/out"
$T --steps 200 --backend furiosa:auto,furiosa:auto --resident-layers 8 --run-name ntc-r200-res8            # Job 1 (카드 2장)
$T --steps 25 --backend furiosa:auto --run-name ntc-t1c-off; \
$T --steps 25 --backend furiosa:auto --resident-layers 4 --run-name ntc-t1c-res4                           # Job 2 (카드 1장)
```

## Results

### 1. 200 step — 2장 × 카드당 상주 8 (#40)

**200/200 step, OOM 없음.** 실패·재시도 없이 한 번에 끝났다(`run200.log`).

| 구간 | 평균 loss | step 중앙값 |
|---|---:|---:|
| 1–25 | 1.2624 | 4.43 s |
| 26–50 | 1.1724 | 4.40 s |
| 51–100 | 1.1088 | 4.39 s |
| 101–150 | 1.0970 | 4.53 s |
| 151–200 | 1.1506 | 4.44 s |

- step 6–200 중앙값 **4.43 s, 463 tok/s**(대기 0.23 s, 상주 갱신 0.47 s = 29.4 ms/층). 6–25 중앙값 4.42 s 로 200 step 내내 같고, 가장 느린 step 도 4.92 s 다 —
  step 시간이 늘거나 메모리가 쌓이는 기색이 없다.
- loss: step 1–5 1.7202, 1.5635, 1.0860, 1.1059, 1.3381. 구간 평균은 51–150 에서 가장 낮고 끝 구간이 조금 오른다(cosine 으로 lr 이 0 에 가까워지는 구간,
  예제마다 loss 차가 커서 50 step 평균도 ±0.05 흔들린다). 품질 평가는 이 기록의 범위가 아니다.
- 10-21 같은 구성 25 step(#42 이전 head)은 4.27 s 였다. #43 이후 head 경로가 달라져 4 % 느려진 것으로 보지만, 같은 시각 2장 오프로드는 이번에 돌리지 않아
  (카드 2장을 한 번만 쓰기로 함) 직접 나누지 못했다.

### 2. 1장 × 상주 4, 경합 없는 host (#34)

6–25 step 중앙값. 두 실행은 같은 Job 에서 이어서 돌았다.

| 실행 | load1 | step | tok/s | 대기 | D2H 합 | AdamW 합 | H2D 합 | 상주 갱신 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1장 오프로드 (`ntc-t1c-off`) | 10–44 | 5.11 s | 401 | 1.59 | 2.38 | 4.16 | 2.45 | — |
| 1장 상주 4 (`ntc-t1c-res4`) | 10–21 | **4.79 s** | **427** | 1.23 | 2.25 | 3.88 | 2.24 | 0.16 |

- 1장에서 상주 4 층은 오프로드보다 **−6 %(+7 % tok/s)**. 2장 × 상주 8 은 1장 오프로드보다 −13 %(4.43 대 5.11 s, +16 % tok/s).
- 25 step 모두 OOM 없음. 10-21 의 1장 상주 4 중 경합 없는 값은 OOM 으로 6–12 step 만 있었다(5.12 s); 이번 값이 그것을 대신한다.

**step 모델** (docs/design.md; 같은 Job 의 1장 오프로드 항: `T_dev` = 5.11 − 1.59 = 3.53 s, `t_a` = 4.16/32 = 0.130 s, `t_u` = 2.45/32 = 0.077 s, W = 4,
장치 AdamW 28 ms/층, `t_fwd` = T_dev/384):

| 구성 | T_dev | T_res | T_wait | 예측 | 측정 | 오차 |
|---|---:|---:|---:|---:|---:|---:|
| 1장 상주 4 | 3.53 | 0.11 | 28 × 0.207 / 4 − 0.11 − 4 × 0.0092 = 1.30 | **4.94 s** | 4.79 s | **+3.0 %** |
| 2장 상주 8 (T_dev 는 10-21 2장 오프로드 3.47 s, host 항은 위 1장) | 3.47 | 0.45 | 0.31 | 4.23 s | 4.42 s (6–25) | −4.4 % |

2장 줄의 `T_dev` 는 #43 이전 코드의 값이라 참고용이다. 장치 AdamW 는 1장 39.5 ms/층, 2장 29.6 ms/층으로 쟀다(모델은 28 ms).

Raw: `experiments/data/resident-200/r200.jsonl`(200 step), `t1c.jsonl`(`ntc-t1c-off`·`ntc-t1c-res4`), `run200.log`(제출 전 확인·결과),
`load-npu-node-1.txt`(60 s 간격 load).

## Conclusion

#42 의 잠금으로 2장 × 카드당 상주 8 이 200 step 을 OOM 없이 끝냈다(4.43 s/step, 463 tok/s, step 시간 변화 없음) — #40 의 완료 조건이다.
경합 없는 host 에서 1장 × 상주 4 는 4.79 s/step 으로 같은 시각 1장 오프로드(5.11 s)보다 6 % 빠르고, host 처리량 step 모델이 +3.0 % 로 맞췄다.
