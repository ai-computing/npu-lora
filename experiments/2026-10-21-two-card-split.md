# 8B 전체 파인튜닝을 RNGD 2장에 레이어 분할: 1장 + 오프로드 대비 step 시간, 옵티마이저 배치 모델 확인

- **Date**: 2026-10-21
- **Task**: #25 (NPUTRAIN-18)
- **Host / device**: npu-node-2 RNGD 2장(npu0, npu1; 한 pod 가 `furiosa.ai/rngd: 2` 로 둘 다 배정받음), 노드 CPU 24코어,
  `furiosaai/furiosa-llm:latest`(digest `sha256:37a85288…`, furiosa-torch 2026.2.0), 드라이버·펌웨어 2026.4.0.
- **Commit**: 2장 `9fbb411`(exp/25-two-card-split: `--backend furiosa:0,furiosa:1`, `--resident-layers`), 1장 기준 `b3a5e90`
  (#22/#23 의 RNGD full FT 실행과 같은 실행).

## Question

1. 레이어 0–15 를 카드 0, 16–31 을 카드 1 에 두면(카드마다 PCIe 링크 하나) 오프로드가 정하는 step 시간이 줄어드는가?
2. 논문 §5.3 의 옵티마이저 배치 모델(전송 바이트/링크 대역 대 원소별 연산)이 그 step 시간을 예측하는가?
3. 카드에 fp32 master·AdamW 상태를 상주시킬 수 있는 레이어에서 장치 AdamW 가 이득인가?

## Setup

- 구현(이 브랜치): 레이어 i 는 카드 ⌊2i/32⌋, head 는 마지막 카드. 경계에서 hidden(S×H bf16 4 MB)과 그 기울기만 host 를 거쳐 옮긴다.
  레이어 그래프는 카드마다 한 벌(같은 컴파일 캐시). 갱신은 그대로 레이어별 host 작업(4 스레드: 기울기 D2H → AdamW → bf16 H2D)이고,
  작업마다 자기 레이어의 카드를 쓴다. `--resident-layers N` 은 카드마다 앞 N 개 레이어의 fp32 master·m·v 를 카드에 두고 통째 텐서
  AdamW 그래프(`optim_device.ResidentAdamW`)로 갱신한다.
- CPU 검증: `test_two_device_split_and_resident_adamw_equal_one_device` — 4층 모델, CPU 백엔드 2개로 분할(상주 0개, 카드당 1개)한
  결과가 1장 파이프라인과 같다(가중치 상대차 < 1e-2, 상주 레이어는 갱신량 w−w0 기준 < 5e-2; lr 을 2배로 바꾸거나 상태를 갱신하지 않으면 실패).
- 측정: Llama-3.1-8B-Instruct, S=512, accum 4, lr 1e-5, nsmc_mix, seed 0. 2장 25 step, 1장 200 step 중 같은 6–25 step 구간의 중앙값.

## Command

```bash
python3 -m npu_lora.cli train --full --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/quality/nsmc_mix_train.jsonl \
  --backend furiosa:0,furiosa:1 --cache-dir /cache/furiosa --seq-len 512 --steps 25 --accum 4 --lr 1e-5 --seed 0 \
  --run-name w7-split8b-rngd --out /tmp/out-split          # 1장: --backend furiosa:auto (#22 실행)
# 상주 AdamW 그래프 (장치 없이 컴파일 시도)
python3 -m npu_lora.cli compile --full --model meta-llama/Llama-3.1-8B-Instruct --seq-len 512 --accum 4 --resident-layers 1
```

## Results

### 1장 대 2장 (6–25 step 중앙값)

| | step | tok/s | 계산(step − 대기) | 갱신 대기 | D2H 합 | AdamW 합 | H2D 합 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1장 + 오프로드 | 6.36 s | 322 | 3.32 s | 3.04 s | 4.00 s | 7.07 s | 3.19 s |
| **2장 분할 + 오프로드** | **9.26 s** | **221** | 5.76 s | 3.54 s | 5.17 s | 9.71 s | 3.47 s |

(D2H·AdamW·H2D 합은 4 스레드의 레이어 작업 시간을 더한 값.)
값은 같다: 2장의 step 1–5 loss 가 1장과 소수 넷째 자리까지 같다(1.7157, 1.5626, 1.0830, 1.1074, 1.3380).

### 모델 예측 대 측정

- §5.3 모델은 갱신 비용 = 전송 바이트/링크 대역 + 원소별 AdamW 로 본다. 1장 측정에서 전송(D2H+H2D)이 레이어 작업의 50 %(7.19/14.26 s)이므로,
  링크가 두 개가 되면 레이어 작업 10.7 s, 대기 3.04 × 10.7/14.26 = 2.3 s, step ≈ 3.3 + 2.3 = **5.6 s(366 tok/s, +14 %) 예측**.
- **측정은 9.26 s(221 tok/s, −31 %)로 예측과 반대다.** 이유:
  1. 상주 host 버퍼 이후 전송은 링크가 아니라 host 쪽 복사가 정한다(1장 D2H 4.0 s 가 2장에서 5.2 s 로 늘었다 — 링크가 둘이어도 host 메모리
     복사·스레드는 하나). 링크 대역이 병목이라는 모델 전제가 더 이상 맞지 않는다.
  2. host AdamW(24코어)가 가장 큰 항이고, 2장에서 경합으로 7.1 → 9.7 s 로 늘었다.
  3. 장치 계산 경로가 3.3 → 5.8 s 로 느려졌다. 한 sequence 를 두 카드가 차례로 처리하므로 계산이 겹치지 않고, 한 프로세스가 동기 실행(2026.2)으로
     두 장치를 번갈아 부르는 비용과 host 경합이 더해진 것으로 보인다(원인 분해는 하지 않았다).
- 결론적으로 모델에서 바꿔야 할 항은 "링크 대역" → "host 복사·AdamW 처리량"이다. 장치 상주 AdamW 는 바로 이 host 항을 없애므로 여전히 맞는 방향이다.

### 상주 AdamW (질문 3) — 측정하지 못함

- 메모리: 레이어당 fp32 w·m·v 2.62 GB + bf16 가중치 0.44 GB + 누적 기울기 0.44 GB = 3.49 GB. 카드당 16층의 bf16 가중치·기울기(14 GB)와
  head(1 GB) 옆에 상주시킬 수 있는 층은 카드당 약 9개(48 GB 카드, 컴파일 작업 공간 여유 제외)라 32층 전부는 불가하다.
- 통째 텐서 AdamW 그래프 4종([4096,4096], [1024,4096], [14336,4096], [4096,14336])의 컴파일을 장치 없이 시도했으나, 같은 프로세스가 먼저
  8B S=512 학습 그래프 전체를 다시 빌드·검증하느라 CPU 노드(요청 8코어, 공유)에서 2.5시간 넘게 상주 그래프 단계에 이르지 못해 중단했다.
  그래서 `--resident-layers` 의 RNGD 실행 시간은 없다. 기능은 CPU 테스트로만 검증됐다.
- 예측(미측정): 카드당 8층 상주 시 host 레이어 작업이 절반(≈7 s)으로 줄고, 장치 AdamW 는 10-19 측정 28 ms/층 × 16층 ≈ 0.45 s 를 주 스레드에 더한다.

Raw: `experiments/data/w7/split8b-rngd-2card.jsonl`(2장), `experiments/data/w7/energy-full8b-rngd.jsonl`(1장),
`experiments/data/w7/split8b-rngd-2card-err-before-fix.txt`(첫 실행: 경계 레이어 입력이 카드 0 에 남아 역전파가 실패 — 9fbb411 에서 수정).

## Conclusion

레이어 분할 2장은 동작하고 값도 1장과 같지만, 상주 host 버퍼를 쓰는 지금은 1장보다 31 % 느리다. 오프로드 경로의 병목이 PCIe 링크에서 host
(복사·AdamW·CPU 경합)로 옮겨 갔기 때문에, 링크 수를 늘리는 분할만으로는 빨라지지 않는다. 논문 §5.3 의 배치 모델은 링크 대역 대신 host 처리량을
비용으로 쓰도록 고쳐야 하고, 다음 측정은 카드당 상주 AdamW(이 브랜치의 `--resident-layers`)로 host 항을 없앤 실행이다 — 상주 그래프를 미리
컴파일할 시간(전용 CPU 노드)이 필요하다.
