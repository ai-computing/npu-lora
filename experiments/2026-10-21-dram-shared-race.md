# 상주 AdamW 실행의 DRAM_SHARED OOM: 원인(프로세스 전역 메모리 종류 플래그 경합)과 수정

- **Date**: 2026-10-21
- **Task**: #40 (원인·수정), #34 (상주 실행 25 step 측정)
- **Host / device**: npu-node-1 RNGD(카드 3장 중 pod 가 배정받은 1–2장), 노드 CPU 96코어, `furiosaai/furiosa-llm:latest`
  (furiosa-torch 2026.2.0, torch 2.10.0+cpu, transformers 5.1.0), 드라이버·펌웨어 2026.4.0.
  실행마다 직전에 `/proc/*/fd` 에서 `/dev/rngd` 를 연 프로세스가 없음을 확인했다(`runall.log`; 한 번은 다른 사용자의 프로세스가
  npu3 를 8분 열고 있어서 닫힐 때까지 기다렸다). npu-node-2 는 쓰지 않았다.
- **Commit**: 브랜치 `fix/40-dram-shared-race`(main `a07aefc` 위). pod 에는 작업 트리의 `npu_lora/` 를 ConfigMap 으로 올렸다.
- **Host load**: §1 의 학습 실행은 다른 사용자의 CPU 작업으로 load 가 350–650(96코어)인 동안 돌아서 step 시간은 신뢰성 확인용이다.
  §2 는 load 가 내려간 뒤(2장 27–81, 1장 27–111) 쟀다. load 는 `data/dram-shared/load-npu-node-1.txt`(60 s 간격).

## Question

1. `failed to allocate … bytes on DRAM_SHARED` 는 무엇이 차서 나는가? DRAM_SHARED 는 DRAM 과 다른 영역인가, 크기는?
2. 고칠 수 있는가? 고친 뒤 상주 실행(2장 × 상주 8, 1장 × 상주 4)이 25 step 을 3번 연속 끝내는가?

## Setup

가설을 싼 것부터 시험했다.

1. **영역 크기** (`data/dram-shared/pool_probe.py`, 카드 1장): 64 MB bf16 업로드를 계속 쌓으면 47.44 GiB 에서
   `… on DRAM` 으로 멈춘다. 그래프 출력(x·2, 64 MB)을 쌓아도 47.38 GiB 에서 `DRAM`. 업로드 8 GiB 를 잡고 그래프 출력을 쌓으면
   39.38 GiB 에서 멈춘다 — 업로드와 그래프 출력은 같은 DRAM 한 영역(47.5 GiB, `capacity: 51002736640`)에 잡힌다.
   DRAM_SHARED 에는 아무것도 잡히지 않는다. 실패 메시지의 `capacity: 0` 이 DRAM_SHARED 영역 크기다(칩 1개 카드).
2. **누가 DRAM_SHARED 를 쓰는가**: `furiosa.native_torch.memory` 에 `_set_dram_shared_mode()` / `_set_memory_type()` 이 있고,
   `furiosa.torch.utils.with_dram_shared` 가 이것으로 할당 종류를 잠깐 바꾼다. 이 플래그가 스레드별인지 시험했다
   (`data/dram-shared/race_repro.py`).
   - `flag`: 다른 스레드에서 켠 DRAM_SHARED 모드를 메인 스레드도 본다(`True`). 그 상태에서 메인 스레드의 8 MB 업로드는
     `OutOfMemoryError: failed to allocate 8388608 bytes on DRAM_SHARED` — #40 과 같은 오류다. **플래그는 프로세스 전역이다.**
   - `watch`: 메인 스레드가 x·2 그래프를 300번 부르는 동안 다른 스레드가 플래그를 7 994 000번 읽어 1 978번 DRAM_SHARED 를 봤다
     (상주 AdamW 그래프 30번: 1 763 563번 중 72번). **모든 그래프 호출이 플래그를 잠깐 DRAM_SHARED 로 바꾼다.**
   - `race`: 메인 스레드는 그래프, 4 스레드는 [1024, 4096] bf16 업로드를 반복 → 그래프 6번 만에 DRAM_SHARED OOM.
     상주 AdamW 그래프 [14336, 4096] 와 같이 돌려도 3 s 안에 같은 OOM.
3. **결론 — 원인**: 학습 루프는 메인 스레드에서 그래프를, 레이어 작업 스레드(`--update-workers`, 기본 4)에서 bf16 가중치 업로드를
   돌린다. 업로드의 장치 할당이 다른 스레드의 그래프 호출 순간과 겹치면 DRAM_SHARED(0 바이트)에 잡혀 실패한다. 메모리 부족·누수·조각화가
   아니다. 상주 실행은 step 끝에 메인 스레드가 상주 AdamW 그래프(카드당 상주 층 × 7개)를 연달아 부르는 바로 그때 작업 스레드들이
   업로드를 시작해서 겹침이 많았고, 오프로드만 하는 실행은 메인 스레드가 대부분 업로드 future 를 기다리느라 겹침이 적어 피했다
   (같은 경합이 확률만 낮을 뿐 있다). 상주 층 수·step 수와 상관없이 "몇 step 뒤 간헐적"인 것도 이것으로 설명된다.
4. 기각한 가설: 업로드 스테이징 영역 고갈(업로드는 DRAM 에 직접 잡히고, 한 스레드에서 400회 반복해도 늘지 않았다 — 10-21 기록),
   조각화(DRAM 은 47.4 GiB 까지 채워지고 실패는 다른 영역), 참조로 살아 있는 버퍼(실패 영역이 비어 있다).

**수정** (`npu_lora/backends.py`): `DEVICE_ALLOC_LOCK` 하나를 그래프 호출(`Serialized` 래퍼, `FuriosaBackend.compile` 이 돌려줌)과
장치 할당(`FuriosaBackend.to_device` 의 `torch.empty(device=…)`)이 함께 잡는다. 복사(`copy_`)는 잠금 밖이다.
업로드 속도는 같다(117 MB bf16, 중앙값: `t.to(dev)` 19.4 ms = 6.05 GB/s, `empty` + `copy_` 19.5 ms = 6.01 GB/s, 값 동일).
`race_repro.py fixed`: 같은 경합을 120 s 동안 그래프 58 799번·업로드 60 725번 돌려 오류 0. 상주 AdamW 그래프로도 120 s, 그래프
35 557번·업로드 55 075번, 업로드 검사(500번마다 내려 비교) 불일치 0.

## Command

```bash
# 재현 (카드 1장 pod, PYTHONPATH=npu-train)
python3 experiments/data/dram-shared/race_repro.py flag|watch|race 60|fixed 120
python3 experiments/data/dram-shared/pool_probe.py upload|graph|"mix 8"|graphfirst
# 학습 (10-21 resident-adamw 기록과 같은 레시피)
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --cache-dir /cache/furiosa --seq-len 512 --steps 25 --accum 4 --lr 1e-5 --seed 0 --run-name ntc-<run> --out /tmp/out \
  --backend furiosa:auto,furiosa:auto --resident-layers 8     # 1장: --backend furiosa:auto --resident-layers 4
```

## Results

### 1. 신뢰성 — 고친 코드로 연속 실행 (16:26–17:40 KST, host load 350–650)

| 실행 | 카드 | 상주/카드 | step | OOM | step(6–25 중앙값) | tok/s | 대기 | 상주 갱신 | load1 중앙값 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r2-res8-a | 2 | 8 | 25/25 | 0 | 8.13 s | 252 | 0.39 | 0.70 | 501 |
| r2-res8-b | 2 | 8 | 25/25 | 0 | 9.31 s | 220 | 0.49 | 0.68 | 384 |
| r2-res8-c | 2 | 8 | 25/25 | 0 | 9.47 s | 216 | 0.56 | 0.71 | 352 |
| r1-res4-a | 1 | 4 | 25/25 | 0 | 9.61 s | 213 | 2.05 | 0.19 | 384 |
| r1-res4-b | 1 | 4 | 25/25 | 0 | 9.75 s | 210 | 2.13 | 0.15 | 429 |
| r1-res4-c | 1 | 4 | 25/25 | 0 | 12.27 s | 167 | 2.59 | 0.27 | 656 |

- 고치기 전 같은 구성은 2장 × 상주 8 이 5번 중 4번(3, 7, 7, 13 step), 1장 × 상주 4 가 1번 중 1번(12 step) OOM 으로 멈췄다
  (10-21 resident-adamw 기록 §4). 고친 뒤 6번 모두 25 step 을 끝냈다.
- loss 는 고치기 전 상주 실행과 비트 단위로 같다(2장 상주 8: 1.7157, 1.5649, 1.0858, 1.1035, 1.3369; 1장 상주 4: 1.7157, 1.5645,
  1.0821, 1.1063, 1.3412). 같은 구성 세 번도 25 step 모두 같다 — 잠금은 결과를 바꾸지 않는다.
- step 시간은 load 에 따라 8.1–12.3 s 로 흔들린다(host AdamW 합 7.9–11.7 s). 성능 비교와 #34 의 모델 검증은 아래 §2 에서 낮은 load 로 한다.

### 2. 낮은 load 측정 (#34)

load 가 내려가기를 기다려(`timing.log`: 실행 직전 1분 load < 150, 카드 2장이 쿠버네티스에서 비어 있고 열린 카드는 모두 k8s 배정분) 한 Job 에서
2장 상주 8 과 2장 오프로드를 이어서 돌렸고, 이어서 1장 오프로드와 1장 상주 4 를 돌렸다. 6–25 step 중앙값.

| 실행 | step 구간(KST) | load1 | step | tok/s | 대기 | D2H 합 | AdamW 합 | H2D 합 | 상주 갱신 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2장 상주 8 (`ntc-t2-res8`) | 19:38–19:40 | 46–81 | **4.27 s** | 480 | 0.24 | 1.19 | 4.01 | 1.28 | 0.48 |
| 2장 오프로드 (`ntc-t2-off`) | 19:43–19:46 | 27–71 | **5.04 s** | 407 | 1.56 | 2.41 | 4.10 | 2.48 | — |
| 1장 오프로드 (`ntc-t1-off`) | 19:51–19:55 | 71–107 | 10.66 s | 192 | 2.47 | 4.08 | 9.30 | 3.13 | — |
| 1장 상주 4 (`ntc-t1-res4`) | 20:02–20:06 | 27–111 | 9.72 s | 211 | 1.62 | 2.80 | 8.65 | 2.80 | 0.22 |

- **2장 + 카드당 상주 8: 4.27 s/step, 480 tok/s — 같은 시각 2장 오프로드(5.04 s)보다 −15 %(+18 % tok/s).** 25 step 전 구간 값이고,
  OOM 으로 5 step 만 쟀던 10-21 값(4.57 s, steps 3–7)을 대신한다. 오프로드 대기가 1.56 → 0.24 s 로 줄었다.
- 1장 실행 두 개는 load 가 다시 100 근처(96코어)로 올라간 동안 돌았다. host AdamW 합이 9.3 s 로 경합 없을 때(4.18 s, 10-21 기록)의 두 배를
  넘고, 장치 경로(step − 대기)도 8.2 s 로 경합 없을 때(3.7 s)의 두 배다 — 메인 스레드도 CPU 를 기다린다. 그래서 1장 숫자는 경합 조건의 값이다.
  load < 60 에서 1장 두 실행을 다시 하려고 20:10–21:02 동안 10분 간격으로 6번 확인했지만 load 가 84–126 이라 그만뒀다.
- loss 는 상주 실행끼리, 오프로드 실행끼리 §1·10-21 기록과 비트 단위로 같다.

**step 모델 확인** (docs/design.md 「Optimizer placement: step-time model」). 호스트 항은 같은 시각·같은 카드 구성의 오프로드 실행에서 잰다:
`T_dev` = step − 대기, `t_a` = AdamW 합 / 32, `t_u` = H2D 합 / 32, W = 4, 장치 AdamW 28 ms/층, `t_fwd` = T_dev / (32 × 4 × 3).

| 구성 | T_dev | t_a + t_u | T_res | T_wait | 예측 | 측정 | 오차 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2장 상주 8 (2장 오프로드 항, load ~50–70) | 3.47 | 0.128 + 0.077 | 0.45 | 0.30 | **4.22 s** | 4.27 s | **−1.1 %** |
| 1장 상주 4 (1장 오프로드 항, load ~100) | 8.18 | 0.291 + 0.098 | 0.11 | 2.52 | 10.82 s | 9.72 s | +11.3 % |

두 구성 모두 15 % 안이다. 1장 줄은 두 실행 사이에 load 가 흔들려(27–111) 오차가 더 크다. 측정한 장치 AdamW 는 2장 29.8 ms/층, 1장(경합)
54.5 ms/층이다 — 메인 스레드에서 도는 그래프 호출도 host 경합의 영향을 받는다.

Raw: `experiments/data/dram-shared/*.jsonl`(step 기록; `t2.jsonl` 에 `ntc-t2-res8`·`ntc-t2-off` 두 실행), `runall.log`·`timing.log`(실행 전 카드 점유·load 확인과 결과),
`load-npu-node-1.txt`(host load), `race_repro.py`, `pool_probe.py`.

## Conclusion

#40 의 OOM 은 메모리 부족이 아니라 furiosa-torch 2026.2 의 프로세스 전역 메모리 종류 플래그를 그래프 호출과 다른 스레드의 장치 할당이
경합해서 났다. 그래프 호출과 장치 할당을 한 잠금으로 묶어 고쳤고(업로드 속도 변화 없음), 고친 뒤 상주 실행 6번(2장 × 8 세 번,
1장 × 4 세 번, 낮은 load 에서 두 번 더)이 모두 25 step 을 끝냈다.
낮은 load 에서 2장 × 상주 8 은 4.27 s/step(480 tok/s)으로 같은 시각 2장 오프로드(5.04 s)보다 15 % 빠르고, host 처리량 step 모델이
−1.1 % 로 맞췄다. 1장 × 상주 4 는 load ~100 에서만 쟀다(9.72 s, 모델 +11 %). 런타임 쪽 문제(플래그가 스레드별이 아님)는 재현 스크립트와 함께 Furiosa 에 알릴 몫이다.
