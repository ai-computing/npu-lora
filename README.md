# npu-lora

추론 전용으로 나온 AOT 컴파일 NPU(FuriosaAI RNGD)에서 LoRA 파인튜닝을 하는 라이브러리.
CPU·CUDA 백엔드로 같은 코드를 돌려 기준값을 만들고 비교할 수 있다.

## 검증 결과 (2026-10-15, ETRI llm-d 클러스터)

| 항목 | 결과 |
| --- | --- |
| 함수형 Llama vs HF `LlamaForCausalLM` | loss 상대오차 1.7e-5 (q·k·v·o LoRA, 합친 모델 경유) |
| RNGD vs CPU 기울기 (Llama-3.1-8B 2 레이어 + 128k 어휘 head) | loss 1.9e-4, 기울기 cos ≥ 0.9997, 상대오차 ≤ 2.5% |
| RNGD 1장 전체 학습 (Llama-3.1-8B 32 레이어, r=16 q·v, S=128) | 0.7 s/step(약 180 tok/s), 200 step 2분 36초, loss 3.29 → ~0 |
| 학습 전 → 후 평가 (학습에 없는 질문 10개, A100 vLLM 런타임 LoRA) | 0/10 → **10/10** (CUDA 백엔드 학습도 10/10) |
| 첫 컴파일 (레이어 순전파·역전파 + head, 조각 단위) | 약 1.5 시간, 이후 EDF 캐시(15 MB)로 즉시 시작 |

### 품질 검증 — RNGD 학습이 GPU 학습과 같은 품질인가 (2026-10-16)

같은 데이터(NSMC 학습 4,000개 중 1,600개 분량)·시드·설정(S=128, r=16 q·v, lr 2e-4, 400 step × accum 4)으로
RNGD 와 A100 에서 각각 LoRA 학습 → vLLM 오프라인 평가(`npu_lora.bench`).

| | NSMC 시험 정답률 (n=1000, 학습에 없는 리뷰) | MMLU (n=400, 망각 확인) |
| --- | --- | --- |
| 기본 Llama-3.1-8B-Instruct | 79.6% | 60.8% |
| **RNGD 학습 LoRA** | **87.5%** | **63.0%** |
| A100 학습 LoRA (같은 조건) | 87.6% | 63.0% |

loss 곡선(40 step 구간 평균)의 RNGD/A100 상대차 중앙값 2.8%, 마지막 구간 0.0676 vs 0.0667
(`python -m npu_lora.compare a/train.jsonl b/train.jsonl`). 이 검증 과정에서 고친 품질 버그:
Llama 3.x 의 llama3 `rope_scaling` 미적용(HF 대비 1.8e-4 → 3.5e-7), 긴 예제가 응답째 잘리던 문제(이제 프롬프트 쪽을 덜어냄).

재현: `examples/nsmc.py OUT` → `lmd-top --plan train … --set data=…/nsmc_train.jsonl` →
`python -m npu_lora.bench --model … --adapter rngd=… --task nsmc=… --task mmlu=…`.

데모: `examples/cluster_facts.py` 로 만든 데이터를 `lmd-top --plan train … --apply` 로 학습하고
`npu-lora serve` → `npu-lora eval`.

## 왜 NPU 에서 학습이 되나

NPU 컴파일러는 "가중치는 상수"라고 가정하고 순전파만 컴파일한다. npu-lora 는 이 가정을 피한다.

1. 고정 가중치와 LoRA 어댑터(A, B)를 **모두 그래프 입력**으로 둔다. 레이어 하나의 그래프를 한 번 컴파일해
   32개 레이어에 가중치만 바꿔 재사용한다.
2. `torch.func.vjp` 를 `make_fx` 로 추적해 **역전파가 평범한 연산으로 펼쳐진 그래프**를 만든다.
   NPU 는 그 그래프를 추론 그래프와 똑같이 실행한다.
3. 옵티마이저(AdamW)는 host 에서 fp32 로 돈다. 어댑터가 작아서(Llama-8B, r=16, q·v 에 13.6 MB)
   step 마다 올리고 내리는 비용이 작다.

장치가 실행하는 그래프: 레이어 순전파, 레이어 역전파(순전파 재계산 = 레이어 단위 activation
checkpointing), head 의 rms·어휘 조각 통계·어휘 조각 기울기·rms 역전파. 각각 행렬곱 2개 단위 조각으로
컴파일되고(Llama-8B 레이어: 순전파 8조각, 역전파 19조각) 32 레이어·모든 어휘 조각에서 재사용된다.

## 컴파일러 우회 (furiosa-torch 2026.2)

| 증상 | 원인 | 대응 |
| --- | --- | --- |
| `UnsupportedOpError` / global-compiler `zip_eq` 패닉 | 그래프 안의 0차원 텐서(스칼라 loss) | loss 를 `[1,1]` 로 두고 `grad` 대신 `vjp` |
| 같은 패닉 | reduction 역전파가 만든 stride-0 `expand` 가 원소별 연산을 두 번 거친 뒤 곱해짐 | `graph.drop_broadcast_expands`: 원소별 연산에만 쓰이는 expand 제거(CPU 재실행으로 검증) |
| 큰 그래프가 컴파일되지 않거나 느림 | `max_num_unique_shapes` 무제한(기본값) | `--max-shapes 8` (기본) |
| cross-entropy 의 gather/scatter | 정수 label | 장치에서 `iota == labels` 로 one-hot |
| `memory explosion` (RMSNorm 역전파 + SwiGLU 역전파), attention 순전파 15분+ | 큰 융합 그래프 | `graph.compile_parts`: 행렬곱 2개마다 그래프를 잘라 조각별 컴파일(`--max-mm`), 조각 사이는 장치 텐서 |
| MLP 역전파 `UnsupportedOpError` | SiLU 역전파의 `empty_like`+`fill_` 상수 텐서, in-place 연산 | `functionalize` + `scalarize_constants` (`1 - s` → `-s + 1`) |
| RoPE·GQA 조각 컴파일 20분+ | slice/neg/cat, repeat_interleave(expand+clone) 재배치 | RoPE 를 `x @ R`(고정 회전행렬)로, GQA 를 q 그룹 reshape + bmm 으로 |
| `verification of operator failed` (Elementwise) | 원소별 연산 두 피연산자의 메모리 배치가 다름(전치한 q 에 RoPE) | RoPE 를 전치 전 `[S, heads, hd]` 에서 |
| CUDA 백엔드에서 CPU/GPU 텐서 섞임 | CPU 샘플로 추적할 때 `ones_like(device=cpu)` 가 박힘 | `strip_device_kwargs` |
| softmax 역전파 `verification of operator failed (EinsumByDpe)` | softmax 출력이 조각 입력일 때 mul+reduce 융합 | `decompose_softmax_backward`: `y·(g − mean(g·y)·N)` |
| `multiple internal subgraphs are not supported` (head) | [S, 128k] fp32 logits 가 온칩 메모리를 넘음 | 어휘 조각 cross-entropy: 조각마다 (max, Σexp, 정답 logit) → host 에서 logsumexp → 조각별 `(softmax−onehot)·W` 누적 |
| 그래프가 조용히 틀린 값을 냄 | 같은 텐서 객체를 두 샘플 자리에 넘기면 make_fx 가 두 입력을 하나로 묶음 | `graph.build` 가 중복 샘플을 복제 |
| **조각 출력이 조용히 전치됨**(정사각이면 rel 1.41, 직사각이면 모양 뒤바뀜) | 컴파일 결과가 출력을 연속 배치로 가정 — 전치 view·전치 텐서의 원소별 결과가 출력이면 틀림 | `materialize_outputs`: 연속이 아닌 출력 앞에 `clone(contiguous)` |
| **연쇄 실행에서만 조용히 틀림**(조각 단독은 정확) | 커널이 컴파일 샘플 입력의 메모리 배치(전치 stride)로 굳음 — 실행 때 연속 입력을 받으면 엉뚱하게 읽음 | 컴파일 샘플을 `contiguous()` 로 맞추고, 캐시 키에 입력 stride 포함 |
| `npu-lora check` 가 엉뚱한 차이 | furiosa 를 불러오면 `torch.manual_seed` 가 장치 쪽에서 경고만 내고 끝나 두 trainer 의 A 초기값이 달라짐 | 어댑터 초기화에 전용 `torch.Generator` |

값이 틀리는 문제는 컴파일 오류가 나지 않는다. `npu_lora.diagnose.compare_parts` 로 조각마다 CPU 와 장치 출력을
같은 입력으로 비교한다. 학습 루프의 컴파일 샘플은 0/상수라서 이런 문제를 잡지 못하니, 새 모델은 반드시
`npu-lora check` 로 기울기를 비교한다.

새 모델·새 SDK 에서 컴파일이 실패하면 `npu_lora.diagnose.first_failing_node` 로 처음 실패하는 노드를 찾고,
의미를 바꾸지 않는 변환을 `graph.PASSES` 에 넣는다. 모든 변환은 `graph.build` 가 CPU 재실행으로 결과가
같은지 확인한다.

## 사용

```bash
# 1) NPU 기울기가 CPU 와 같은지 (앞 2개 레이어)
npu-lora check --model meta-llama/Llama-3.1-8B-Instruct --backend furiosa:0 --layers 2

# 2) 학습 — JSONL: {"text"} | {"prompt","response"} | {"messages"}
npu-lora train --model meta-llama/Llama-3.1-8B-Instruct --data data.jsonl \
    --backend furiosa:0 --steps 200 --out ./adapter --metrics-port 9400

# 3) NPU 서빙용으로 합치기 (Furiosa/RBLN 은 artifact 를 다시 컴파일해 서빙)
npu-lora merge --model meta-llama/Llama-3.1-8B-Instruct --adapter ./adapter --out ./merged

# 4) 학습한 어댑터를 실행 중인 vLLM(--enable-lora, VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)에 올리고 평가
npu-lora serve --endpoint http://<vllm>:8000 --name facts --path /mnt/store/adapters/<model>/<run>
npu-lora eval  --endpoint http://<vllm>:8000 --model facts --data eval.jsonl

# 5) 클러스터에서 — 공식 furiosa-llm 이미지 + 소스 ConfigMap (lmd-top: --plan train 이 같은 Job 을 만든다)
npu-lora job --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/x.jsonl \
    --name demo --node etri-001 --hf-host-path /home/gspark/.cache/huggingface | kubectl apply -f -
```

출력은 HF PEFT 형식(`adapter_config.json`, `adapter_model.safetensors`)이라 vLLM `--enable-lora`
와 llm-d 의 LoRA 라우팅이 그대로 읽는다. `npu_lora_meta.json` 에 학습 설정·백엔드·SDK 버전이 남는다.

## 관측

- stdout(=`kubectl logs`)과 `--log-file` 에 JSONL: `start`, `log`(가중치 적재·컴파일), `step`, `done`.
- `--metrics-port` 를 주면 Prometheus `/metrics`: `npu_lora_loss`, `npu_lora_tokens_per_second`,
  `npu_lora_step`, `npu_lora_steps_planned`, `npu_lora_tokens_total` …
  `npu-lora job` 이 ServiceMonitor 를 함께 만들어 클러스터 Prometheus 가 수집한다.

## 모듈

| 파일 | 역할 |
| --- | --- |
| `llama.py` | Llama 레이어·head 를 순수 함수로, safetensors 에서 필요한 텐서만 읽기 |
| `graph.py` | 순전파+역전파 그래프 추적과 컴파일러 우회 변환 |
| `backends.py` | furiosa / cuda / cpu 실행과 컴파일 캐시(EDF) |
| `trainer.py` | 레이어별 순전파·역전파, host AdamW, 스케줄 |
| `data.py` | JSONL → 길이 S packing, 응답 토큰만 loss, 문서 경계 마스크 |
| `export.py` | PEFT 저장, 합친 모델 저장 |
| `metrics.py` | JSONL·Prometheus 보고 |
| `k8s.py` | 학습 Job 매니페스트 |
| `diagnose.py` | 첫 실패 노드 이분 탐색, 조각별 CPU/장치 출력 비교 |
| `cli.py` | `train` · `check` · `merge` · `serve` · `eval` · `job` |

## 한계

- 지금은 Llama 계열(GQA, RoPE, SwiGLU)만. 다른 구조는 `llama.py` 처럼 순수 함수로 옮기면 된다.
- micro-batch 는 시퀀스 1개(`--accum` 으로 누적). 시퀀스 길이는 컴파일 시점에 고정된다(새 길이 = 새 컴파일).
- furiosa-torch 는 privileged 컨테이너에서만 장치를 연다. Job 은 init 컨테이너가 배정 카드를 적고
  학습 컨테이너가 그 카드의 서수(`rngd:N` = 보이는 카드 중 N 번째)를 쓴다(`NPU_LORA_RNGD_FILE`).
- 어휘 조각 통계는 장치 수치 오차 때문에 loss 가 0 근처에서 살짝 음수로 보일 수 있다(기울기와 무관).
- RBLN(ATOM)은 같은 방식이 가능하지만(`rebel.compile_from_torch`), 이 클러스터의 RBLN 컴파일 경로가
  고장 난 상태라 백엔드를 아직 넣지 않았다.
