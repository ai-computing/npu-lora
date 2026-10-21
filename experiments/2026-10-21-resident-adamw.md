# 8B 전체 파인튜닝: 카드 상주 AdamW 컴파일·측정과 host 처리량 step 모델

- **Date**: 2026-10-21
- **Task**: #34
- **Host / device**: npu-node-1 RNGD 2장(한 pod 가 `furiosa.ai/rngd: 2` 로 배정받은 두 장; 1장 실행은 그중 한 장), 노드 CPU 96코어,
  `furiosaai/furiosa-llm:latest`(furiosa-torch 2026.2.0, torch 2.10.0+cpu, transformers 5.1.0), 드라이버·펌웨어 2026.4.0.
  실행 전 `/proc/*/fd` 에 `/dev/rngd` 를 연 프로세스가 없음을 확인했다. 컴파일은 장치 없는 CPU 노드(요청 16코어).
  npu-node-2 는 쓰지 않았다.
- **Commit**: 브랜치 `perf/34-resident-adamw-compile`(main `833fef7` 위). pod 에는 작업 트리의 `npu_lora/` 를 ConfigMap 으로 올렸다.
- **주의 — host 경합**: 14:15 KST 무렵부터 같은 노드에서 다른 사용자의 CPU 작업(nice, load ≈ 100/96코어)이 돌았다.
  그 전 실행(아래 "경합 없음")과 그 뒤 실행("경합")은 직접 비교하지 않는다. load 는 `data/resident-adamw/load-npu-node-1.txt`(60 s 간격).

## Question

1. 상주 AdamW 그래프(`--resident-layers N`)가 2.5 시간 안에 컴파일되지 않은 이유는? 고칠 수 있는가?
2. 카드당 상주 레이어 약 8개로 1장·2장 step 시간은? loss 는 오프로드 실행과 같은가?
3. 링크 대역 대신 host 복사·AdamW 처리량으로 쓴 step 모델이 측정을 15 % 안으로 예측하는가?
4. (부가) host AdamW 작업 스레드를 늘리면 빨라지는가?

## Setup

- **컴파일이 느렸던 이유**: `npu-lora compile --resident-layers 1` 은 상주 AdamW 그래프 전에 학습 그래프 경로 전체를 다시 돈다 —
  8B 가중치 읽기, fp32 master 32 GB, 레이어 그래프 3종 추적(`graph.build` 가 CPU 에서 변환 전후를 실행해 비교), 그리고
  `compile_parts` 가 조각마다 CPU 에서 샘플을 실행해 입력 모양을 얻는다. 조각이 모두 캐시에 있어도 이 CPU 실행은 매번 한다.
  10-21 로그(`c-res8b`)에서 순전파 11조각은 캐시에서 0 s 에 읽혔고, 그다음 역전파 조각 실행에서 2시간 넘게 출력이 없었다
  (공유 CPU 노드, 요청 8코어, 같은 노드에 컴파일 작업 여러 개). AdamW 그래프 자체는 가중치 모양에만 의존한다.
- **고침**: `npu-lora compile --adamw-only` — 가중치를 읽지 않고 safetensors 헤더에서 레이어의 2차원 가중치 모양만 얻어
  (`Weights.shape`, `optim_device.resident_shapes`) 모양마다 `ResidentAdamW` 그래프만 컴파일해 캐시에 넣는다. trainer 도 같은 함수로 모양을 정한다.
- **카드 배정**: 카드 3장 노드에서 2장 pod 는 `--backend furiosa:0,furiosa:1` 이 배정받지 않은 카드를 열 수 있었다(서수 0·1 = 노드의 첫 두 장).
  `--backend furiosa:auto,furiosa:auto` 가 init 컨테이너가 적은 배정 카드 목록(`npu1,npu3` 같은)에서 차례로 고르게 했다. 예전 코드는
  이 목록을 숫자 하나로 읽었다(`npu1,npu3` → 13).
- **메모리**: 상주 갱신이 새 bf16 가중치를 만들기 전에 옛 장치 가중치를 놓고(`_layer_job` 과 같게), 끝난 레이어의 누적 기울기를 바로 놓는다.
- `--update-workers N`(기본 4): 레이어별 host 작업(기울기 D2H → AdamW → bf16 H2D) 스레드 수.
- 측정: Llama-3.1-8B-Instruct, S=512, accum 4, lr 1e-5, nsmc_mix, seed 0 — two-card 기록과 같은 레시피. 25 step, 6–25 step 중앙값
  (OOM 으로 일찍 멈춘 실행은 남은 구간, 표에 적었다).

## Command

```bash
# 상주 AdamW 그래프만 (장치 없는 CPU 노드)
python3 -m npu_lora.cli compile --adamw-only --full --model meta-llama/Llama-3.1-8B-Instruct --seq-len 512 --cache-dir /cache/furiosa
# 학습 (RNGD pod; 2장은 furiosa.ai/rngd: 2 와 --backend furiosa:auto,furiosa:auto)
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --cache-dir /cache/furiosa --seq-len 512 --steps 25 --accum 4 --lr 1e-5 --seed 0 --run-name ntb-<run> --out /tmp/out \
  --backend furiosa:auto [--resident-layers N] [--update-workers 8]
```

Job 생성기는 `experiments/data/w7/jobgen.py` 와 같다(라벨·로그 경로만 바꿈).

## Results

### 1. 컴파일 — 2.5 h 이상 → 150 s

| 그래프 | 모양 | 컴파일 |
|---|---|---:|
| `adamw-res-fp32` | 1024×4096 (k, v) | 4 s |
| | 4096×4096 (q, o) | 15 s |
| | 4096×14336 (down) | 58 s |
| | 14336×4096 (gate, up) | 73 s |
| 합 | | **150 s** |

학습 실행은 이 캐시를 그대로 읽는다(`[furiosa] adamw-res-fp32-…: cache 0s`).

### 2. step 시간 (경합 없음, 13:15–14:11 KST)

| 실행 | 카드 | 상주/카드 | 구간 | step | tok/s | 대기 | D2H 합 | AdamW 합 | H2D 합 | 상주 갱신 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 오프로드 | 1 | 0 | 6–25 | **5.22 s** | 392 | 1.50 | 2.22 | 4.18 | 2.27 | — |
| 오프로드, `--update-workers 8` | 1 | 0 | 6–25 | **4.82 s** | 425 | 1.23 | 3.21 | 8.05 | 3.62 | — |
| 오프로드 | 2 | 0 | 6–25 | **5.09 s** | 402 | 1.50 | 2.21 | 4.21 | 2.30 | — |
| 상주 | 2 | 8 | 3–7 (7 에서 OOM) | **4.57 s** | 448 | 0.15 | 1.12 | 4.19 | 1.19 | 0.52 |
| 상주 | 1 | 4 | 6–12 (12 에서 OOM) | **5.12 s** | 400 | 1.04 | 1.93 | 4.22 | 1.96 | 0.18 |
| 상주 | 1 | 8 | — | 첫 step 에서 OOM(카드 1장에 약 50 GB, 용량 초과) | | | | | | |

(합은 레이어 작업 시간을 스레드에 걸쳐 더한 값. 상주 레이어는 host 작업이 없어서 2장 상주 실행의 합은 오프로드 16층 몫이다.)

- **2장 + 카드당 8층 상주: 4.57 s/step, 1장 오프로드(5.22 s) 대비 +14 % tok/s.** 대기가 1.50 → 0.15 s 로 거의 사라졌고,
  step 은 장치 계산(약 3.6 s)과 상주 갱신(0.52 s, 16층 × 33 ms)이 정한다. 단 OOM 으로 5 step 만 쟀다(아래).
- 이 96코어 host 에서는 2장 오프로드가 1장과 같다(5.09 대 5.22 s). 24코어 npu-node-2 에서 2장이 31 % 느렸던 것(two-card 기록)과 다르다 —
  host 처리량이 충분하면 분할 자체의 손해는 거의 없다.
- **host 개선: 작업 스레드 4 → 8 로 −8 %(5.22 → 4.82 s).** 레이어당 AdamW 는 경합으로 0.131 → 0.252 s 로 늘지만 병렬도가 두 배라 대기가 줄었다.

### 3. loss — 오프로드와 같은가

| step | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 오프로드(1장·2장, 스레드 4·8 모두 같음) | 1.7157 | 1.5626 | 1.0879 | 1.1060 | 1.3361 |
| 2장 상주 8 | 1.7157 | 1.5649 | 1.0858 | 1.1035 | 1.3369 |
| 1장 상주 4 | 1.7157 | 1.5645 | 1.0821 | 1.1063 | 1.3412 |

오프로드 실행끼리는 비트 단위로 같고, 상주 실행은 step 2 부터 최대 0.5 %(step 5, 상주 4) 다르다. 장치 AdamW 는 host fused AdamW 와 갱신량 기준
1e-5 수준으로 같지만(10-19 기록), warmup 초반 lr 1e-6 에서 갱신이 bf16 가중치 한 ulp 보다 작은 원소가 대부분이라 fp32 차이가 bf16 반올림을
바꾼 원소들이 loss 를 움직인 것으로 본다(검증은 하지 않았다). 같은 코드의 CPU 테스트(`test_two_device_split_and_resident_adamw_equal_one_device`)는
갱신량 상대차 < 5e-2 로 통과한다.

### 4. 상주 실행의 장치 OOM (#40)

상주 실행은 계획 메모리가 용량(47.5 GiB) 아래인데도 몇 step 뒤 `failed to allocate … bytes on DRAM_SHARED` 로 멈췄다. 실패 위치는 레이어 작업 스레드의
bf16 업로드(`to_device`)다.

| 실행 | 계획 메모리/카드 | 멈춘 step |
|---|---:|---:|
| 2장 상주 8 (옛 가중치 놓기 전) | ~36 GB | 3 |
| 2장 상주 8 | ~36 GB | 7, 7(경합), 13(경합), 완주 1회(경합) |
| 2장 상주 6 (경합) | ~31 GB | 7 |
| 1장 상주 4 | ~40 GB | 12 |
| 1장 상주 3 (경합) | ~37 GB | 완주 |
| 2장 오프로드 | ~15 GB | 완주(2회) |

- 각 조각을 따로 반복하면 늘지 않는다: [14336, 4096] 상주 AdamW 그래프 400회(상태 교체), 같은 모양 bf16 업로드 400회, [1,1] 스칼라 업로드 2만 회 모두 OOM 없음.
- `furiosa-smi status` 는 실행 내내 0.00/47.50 GiB 로 보여서 밖에서 메모리를 볼 수 없었다.
- 상주 갱신을 오프로드 업로드보다 먼저 돌리는 순서(조각화 가설)도 시험했지만 결과가 섞여(2장 상주 8: 13, 완주; 2장 상주 6: 7) 되돌렸다.
  `r2-res8-try3`, `r2-res8-mem`, `r2-res6`, `r1-res3` 이 그 변형이고, 나머지와 PR 코드는 오프로드 업로드와 상주 갱신이 겹치는 원래 순서다.
  원인은 모른다 — #40 에서 따로 본다.

### 5. 경합 host 에서 같은 조건 비교 (15:00–15:25 KST)

host load 가 약 100(96코어)인 동안 PR 코드로 2장 상주 8 과 2장 오프로드를 이어서 돌렸다.

| 실행 | 구간 | step | tok/s | 대기 | AdamW 합 | H2D 합 | 상주 갱신 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2장 오프로드 | 6–25 | 10.96 s | 187 | 2.66 | 9.27 | 3.39 | — |
| 2장 상주 8 | 6–7 (7 에서 OOM) | 8.83 s | 232 | 0.27 | 8.67 | 1.54 | 0.84 |

경합 host 에서도 상주가 대기를 2.66 → 0.27 s 로 없애고 step −19 %(+24 % tok/s)다. 레이어당 host AdamW 가 경합 없을 때의 2–4배(0.29 s)라
host 항을 없애는 이득이 더 크다. 이 조건의 항(T_dev 8.30 s, t_a + t_u 0.396 s)으로 모델은 9.71 s 를 예측한다(측정 8.83 s, +10 %).
같은 경합 조건에서 이보다 앞서 다른 순서(상주 갱신 먼저) 변형이 25 step 을 완주했다: 9.48 s/step, 216 tok/s(`r2-res8-mem.jsonl`).

### 6. step 모델 (docs/design.md 「Optimizer placement: step-time model」)

`T_step ≈ T_dev + T_res + max(0, N_off·(t_a + t_u)/W − T_res − N_res,first·t_fwd)`.
`T_dev` 는 같은 카드 구성의 오프로드 실행에서 step − 대기, `t_a`·`t_u` 는 같은 host 1장 오프로드 실행의 레이어당 host AdamW·업로드
(0.131 + 0.071 s, W=4), 장치 AdamW 는 10-19 측정 28 ms/층, `t_fwd` = T_dev / (32 × accum × 3).
기울기 D2H 는 역전파와 겹쳐 임계 경로에 없다.

| 구성 | 예측 | 측정 | 오차 |
|---|---:|---:|---:|
| 96코어, 1장 오프로드, W=8 (W=4 의 레이어당 값) | 4.53 s | 4.82 s | −6 % |
| 96코어, 1장 상주 4 | 5.10 s | 5.12 s | −0.4 % |
| 96코어, 2장 상주 8 | 4.33 s | 4.57 s | −5 % |
| 24코어(npu-node-2), 1장 오프로드 (대기 항) | 5.89 s | 6.36 s | −7 % |
| 24코어(npu-node-2), 2장 오프로드 | 8.33 s | 9.26 s | −10 % |
| 96코어 경합, 2장 상주 8 (같은 조건 2장 오프로드의 항) | 9.71 s | 8.83 s | +10 % |

모두 15 % 안이다(경합 줄은 2 step 구간이라 잡음이 크다). 링크 대역 모델은 마지막 줄을 5.6 s(−40 %)로 예측했다.
예: 2장 상주 8 은 T_dev 3.60 s(2장 오프로드 5.09 − 1.50) + T_res 16 × 0.028 = 0.45 s + 대기 max(0, 16 × 0.202 / 4 − 0.45 − 8 × 0.0094) = 0.28 s = 4.33 s.

Raw: `experiments/data/resident-adamw/*.jsonl`(step 기록; `*-try*`·`*-oom` 은 OOM 으로 멈춘 실행, `.err` 는 그 오류 줄),
`load-npu-node-1.txt`(host load).

## Conclusion

상주 AdamW 그래프는 학습 그래프 경로를 건너뛰는 `compile --adamw-only` 로 150 s 에 컴파일된다(2.5 h 이상에서). 2장에 카드당 8층을 상주시키면
오프로드 대기가 1.5 → 0.15 s 로 사라져 4.57 s/step(+14 %)이지만, 남은 step 은 장치 계산 경로(3.6 s)가 정하고, 상주 실행은 아직 몇 step 뒤
장치 OOM 이 간헐적으로 난다(#40) — 학습에 쓰려면 그것부터 고쳐야 한다. host 쪽에서는 작업 스레드를 8 로 늘리는 것만으로 −8 % 를 얻었다.
host 복사·AdamW 처리량으로 쓴 step 모델은 측정한 여섯 구성을 −10 ~ +10 % 로 예측한다.
