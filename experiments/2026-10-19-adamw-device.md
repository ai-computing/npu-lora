# AdamW 갱신을 RNGD 그래프로: 정확도와 CPU 갱신·전송 대비 시간

- **Date**: 2026-10-19
- **Task**: NPUTRAIN-34
- **Host / device**: npu-node-1 RNGD 1장(npu0, 다른 프로세스 없음을 `furiosa-smi ps` 로 확인, 끝난 뒤 pod 삭제로 반환),
  `furiosaai/furiosa-llm:latest`(furiosa-torch 2026.2.0, torch 2.10.0+cpu), 드라이버·펌웨어 2026.4.0.
  pod CPU request 16(노드 96코어, torch 스레드 48), 노드는 거의 유휴. npu-node-2 은 cordon 이라 쓰지 않았다.
- **Commit**: f9b7bc5-dirty + 새 파일 `npu_lora/optim_device.py`(브랜치 `adamw-device`).
  pod 에는 작업 트리의 `npu_lora/` 를 그대로 복사했다(미커밋 `backends.py` 포함). 실험 코드는 main 에 있는 API 만 쓴다.

## Question

1. Llama-3.1-8B 디코더 레이어 1개(가중치 9종, 218.1M 파라미터)의 AdamW 갱신을 RNGD 그래프로 컴파일할 수 있는가?
2. CPU `torch.optim.AdamW(fused)` fp32 와 비교한 정확도: (a) 상태·master fp32, (b) bf16 상태. 20 step 연쇄로 오차가 쌓이는가?
3. 레이어 1개 시간: NPU 갱신 / CPU fused AdamW / 상태(w, m, v) H2D·D2H / 지금 방식(기울기 D2H + 가중치 H2D).

## Setup

- 그래프 `optim_device.adamw_fn`: 입력 `(w, m, v, g, lr, β1, β2, eps, wd, 1−β1^t, 1−β2^t)` → `(w', m', v')`,
  fp32 변형은 다음 순전파용 bf16 `w'` 도 낸다. 스칼라는 모두 `[1,1]` 텐서, step 에 따라 바뀌는 `1−β^t` 도 입력이다.
  `1 − x` 는 `-x + 1` 로 써서 상수 텐서·rsub 를 만들지 않았다. `graph.build`(functionalize + `graph.PASSES` +
  CPU 재실행 검증)로 추적하고 `graph.compile_parts(max_mm=0)`(연속 샘플, `materialize_outputs`)로 컴파일했다.
  즉 docs/compiler-workarounds.md 의 기존 우회를 그대로 썼고 새 우회는 필요 없었다.
- 조각: 행렬 7종은 `[1024, 4096]` 조각(레이어당 52개), norm 2종은 `[1, 4096]` — 그래프는 변형마다 모양 2개.
- (a) fp32: w·m·v fp32, g bf16 입력(장치 누적 기울기 형식), 계산 fp32.
  (b) bf16: w·m·v 모두 bf16, 내부 계산 fp32 후 bf16 으로 저장. master 없음.
- 데이터: 가중치는 실제 Llama-3.1-8B-Instruct layer 0. 기울기는 **합성**이다: 원소마다 고정 신호 N(0, 0.5e-4) + step 마다
  잡음 N(0, 1e-4), bf16 으로 반올림해 모든 변형과 CPU 기준이 같은 값을 받는다.
  하이퍼파라미터는 8B full FT 와 같은 lr 1e-5, β (0.9, 0.999), eps 1e-8. wd 는 갱신식의 감쇠 항을 확인하려고 0.01
  (trainer 기본값은 0).
- 기준: CPU `torch.optim.AdamW(fused=True)` fp32. 장치 오차와 bf16 형식 오차를 가르려고 같은 bf16 그래프를 CPU 에서도 돌렸다.
- 시간은 7회 중앙값(첫 회 워밍업 제외). 장치 텐서는 이미 장치에 있는 상태에서 잰다. D2H 는 미리 잡은 host 버퍼에
  `copy_`(trainer 의 상주 버퍼 방식), H2D 는 `.to(dev)`.

## Command

```bash
# npu-node-1, RNGD 1장 pod (furiosa-llm:latest, privileged, init 컨테이너가 배정 카드 기록)
python3 -m npu_lora.optim_device --backend cpu --steps 2 --reps 2 --out cpu-smoke.json            # CPU 확인
python3 -m npu_lora.optim_device --backend furiosa:auto --cache-dir /cache/furiosa \
    --steps 20 --reps 7 --out rngd.json
python3 check_timing.py   # 동기화 확인 + CPU AdamW 스레드 수별 (experiments/data/adamw-device-check_timing.py)
```

## Results

### 1. 컴파일

| 변형 | 조각 모양 | aten 연산 수 | 컴파일 | compare_parts 상대오차(출력별) |
|---|---|---:|---:|---|
| fp32 | 1×4096 | 25 | 1.5 s | 1.1e-9, 0, 0, 0 |
| fp32 | 1024×4096 | 25 | 3.0 s | 1.4e-9, 0, 0, 0 |
| bf16 | 1×4096 | 30 | 1.6 s | 0, 0, 0 |
| bf16 | 1024×4096 | 30 | 3.3 s | 0, 0, 0 |

**컴파일된다.** 한 조각 그래프로 레이어 전체(54회 호출)를 돈다. 막힌 노드는 없었다(`first_failing_node` 를 부를 일이 없었다).

### 2. 정확도 (레이어 전체, CPU fused AdamW fp32 기준)

`dw_cum` = 누적 갱신량 (w_t − w_0) 의 상대오차. w 자체 오차는 가중치 크기에 묻히므로 이것이 의미 있는 지표다.

| step | (a) fp32 w | (a) m | (a) v | (a) **dw_cum** | (b) bf16 w | (b) **dw_cum** | 갱신량/가중치 < 2^-8 | < 2^-9 | bf16 w 그대로인 원소 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.9e-8 | 2.2e-7 | 1.3e-5 | **2.4e-5** | 6.7e-4 | 0.862 | 79.2% | 62.7% | 70.2% |
| 10 | 4.9e-8 | 1.6e-7 | 1.3e-5 | **1.2e-5** | 3.4e-3 | 0.861 | 90.3% | 82.6% | 86.9% |
| 20 | 7.1e-8 | 1.3e-7 | 1.3e-5 | **1.0e-5** | 6.1e-3 | 0.865 | 90.9% | 83.7% | 87.8% |

- (a) fp32 장치 갱신은 CPU fused AdamW 와 갱신량 기준 1e-5 수준으로 같고 **20 step 동안 쌓이지 않는다**(오히려 줄어든다).
  같은 그래프를 CPU 에서 돌린 값(smoke run)과 dw_cum 이 유효숫자까지 같다. 남은 1e-5 는 장치 오차가 아니라 lerp 대 mul-add 같은 식 차이다.
  텐서별 dw_cum(step 20): 8.5e-6 ~ 3.3e-5(norm 이 가장 크다). bf16 w' 출력은 w' 를 host 에서 bf16 으로 바꾼 값과 정확히 같다.
- (b) bf16 상태는 **갱신의 86% 를 잃는다**(dw_cum 0.86). lr 1e-5 에서 원소의 91% 는 한 step 갱신량이 가중치의 2^-8 보다 작고,
  실제로 88% 원소는 bf16 가중치가 그대로였다. 같은 bf16 그래프의 CPU 결과와 장치 결과의 차이는 3.2e-7 이므로 이것은
  장치 문제가 아니라 **bf16 형식 자체의 한계**다. bf16 상태는 쓰면 안 된다(fp32 master 또는 Kahan/확률적 반올림이 필요).
  더 큰 lr(예: LoRA 의 2e-4)에서 손실 비율은 측정하지 않았다.

### 3. 시간 (레이어 1개 = 218.1M 파라미터, 중앙값)

| 항목 | 옮기는 양 | 시간 | 환산 |
|---|---:|---:|---:|
| NPU 갱신 그래프 fp32 (54 호출) | — | **28.4 ms** (재측정 28.0, 마지막 조각 D2H 로 완료 강제 시 29.7) | |
| NPU 갱신 그래프 bf16 | — | **24.7 ms** (재측정 24.1 / 24.8) | |
| CPU fused AdamW fp32 | — | **12.5 ms** (48 스레드; 재측정 4/16/48 스레드 26.1 / 17.9 / 18.9 ms) | |
| 상태 w·m·v fp32 H2D | 2.62 GB | 465 ms | 5.6 GB/s |
| 상태 w·m·v fp32 D2H | 2.62 GB | 387 ms | 6.8 GB/s |
| 상태 w·m·v bf16 H2D | 1.31 GB | 218 ms | 6.0 GB/s |
| 상태 w·m·v bf16 D2H | 1.31 GB | 182 ms | 7.2 GB/s |
| 지금 방식: 기울기 bf16 D2H(상주 버퍼) | 436 MB | 62 ms | 7.0 GB/s |
| 지금 방식: 가중치 bf16 H2D | 436 MB | 65 ms | 6.7 GB/s |
| (참고) 기울기 D2H `t.cpu()` | 436 MB | 182 ms | 2.4 GB/s |

레이어당 합(직렬 기준):

| 방식 | 구성 | 레이어당 | ×32 레이어(**추정**, 단순 곱) |
|---|---|---:|---:|
| 지금: CPU 갱신 | 기울기 D2H + CPU AdamW + 가중치 H2D | **0.139 s** | ~4.5 s |
| NPU 갱신, fp32 상태 스트리밍 | 상태 H2D + NPU 갱신 + 상태 D2H | **0.880 s** | ~28 s |
| NPU 갱신, bf16 상태 스트리밍 | 같음(bf16) | 0.425 s | ~14 s (정확도 불가) |

- 갱신 계산 자체는 NPU 와 CPU 가 같은 자릿수다(NPU 24–28 ms, 이 유휴 노드 CPU 13–26 ms). **시간을 정하는 것은 전송량**이다:
  MoE offload 식으로 상태를 가져와 갱신하면 레이어당 2.6 GB × 2 를 옮겨야 해서, 기울기·가중치 0.44 GB × 2 를 옮기는 지금 방식보다 6.3배 느리다.
- 링크 대역은 6–7 GB/s 로 양방향 비슷하다. 상태를 장치에 두는 방식만 의미가 있는데, 8B fp32 상태(w·m·v 7.0B × 12 B ≈ 84 GB)는
  카드 1장(48 GB)에 들어가지 않는다. 카드 여러 장에 레이어를 나눠 상태를 상주시키면(NPUTRAIN-18) 갱신 28 ms × 레이어 수만 남고
  전송이 사라진다(추정, 미측정).
- 이 노드의 CPU AdamW(13–26 ms/레이어)는 e11 8B 학습 중 측정(레이어 작업 합 AdamW 7.1 s / 32 = 0.22 s, 4 스레드 경합, host load 높음)보다
  훨씬 빠르다. 노드·부하가 달라 직접 비교할 수 없다.

Raw logs: `experiments/data/adamw-device-rngd.json`(전체 결과·step 별 연쇄), `experiments/data/adamw-device-rngd.log`,
`experiments/data/adamw-device-check_timing.json` + `.py`(동기화·스레드 확인), `experiments/data/adamw-device-cpu-smoke.json`(CPU 백엔드).

## Conclusion

AdamW 갱신은 RNGD 에서 기존 우회만으로 바로 컴파일되고(조각당 3 s), fp32 상태면 CPU fused AdamW 와 갱신량 1e-5 수준으로 일치하며
20 step 동안 오차가 쌓이지 않는다. bf16 상태는 lr 1e-5 에서 갱신의 86% 를 잃어 쓸 수 없다.
속도에서는 상태를 NPU 로 스트리밍하는 방식이 레이어당 0.88 s 로, 지금의 CPU 갱신 경로(0.14 s)보다 6배 느리다. 갱신 계산은 둘 다
수십 ms 이고 전송량이 전부를 정한다. 그러므로 8B 카드 1장에서는 CPU 갱신을 유지하고, NPU 갱신은 상태가 장치에 상주할 수 있을 때
(카드 여러 장에 레이어 분할, NPUTRAIN-18) 쓴다. 그때는 기울기 D2H·가중치 H2D 도 없어진다.
다음: 다중 카드에 상태를 상주시킨 설계에서 레이어당 갱신을 trainer 에 붙여 step 시간 측정; 합성이 아닌 실제 기울기로 한 번 확인.
