# 컴파일러 우회 (furiosa-torch 2026.2)

추론용 AOT 컴파일러에 역전파 그래프를 넣으면서 만난 문제와 대응이다. 모든 대응은 `npu_train/graph.py` 의
그래프 변환(`graph.PASSES`) 또는 분할·컴파일 단계에 있고, `graph.build` 가 변환 전후 결과를 CPU 로 비교한다.

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
| **fp32 그래프인데 bf16 정밀도**(Qwen2 q LoRA 기울기 RNGD 대 fp32 40%) | RNGD 행렬곱은 그래프가 fp32 여도 입력을 bf16 으로 읽음. Qwen2 의 큰 q/k bias 에서 `dq = dS·k` 가 상쇄(Σ_j dS_ij = 0)되며 오차가 커짐 | q/k 는 bias 덧셈부터 q·kᵀ 까지 fp32, `k − mean_j(k_j)`(softmax 불변, detach)로 큰 공통 성분을 행렬곱 전에 제거 (#33) |
| head 의 정답 logit 이 행 최댓값보다 큼 → 음수 loss | `(logits·onehot).sum` 은 fp32 logits 를 줘도 bf16 로 반올림한 값으로 계산(최대 0.12), max·Σexp 는 fp32 | 조각 그래프는 (max, Σexp)만, 정답 logit 은 host 에서 `target_logits`(S 개 내적) (#33) |
| `UnsupportedOpError` (ReduceByVe verification) | 위치 축 mean 의 역전파(전치 view 위 reduction) | 정확한 기울기가 0 이라 mean 을 `detach` |
| 컴파일은 되는데 실행이 `failed to run TP program … TimedOut` | 전치 view 위에서 위치 축 mean + sub + bmm 이 한 조각에 있을 때 | mean 을 전치 전 `[S, nkv, hd]` 에서 |
| `npu-lora check` 가 엉뚱한 차이 | furiosa 를 불러오면 `torch.manual_seed` 가 장치 쪽에서 경고만 내고 끝나 두 trainer 의 A 초기값이 달라짐 | 어댑터 초기화에 전용 `torch.Generator` |

값이 틀리는 문제는 컴파일 오류가 나지 않는다. `npu_train.diagnose.compare_parts` 로 조각마다 CPU 와 장치 출력을
같은 입력으로 비교한다. 학습 루프의 컴파일 샘플은 0/상수라서 이런 문제를 잡지 못하니, 새 모델은 반드시
`npu-lora check` 로 기울기를 비교한다.

새 모델·새 SDK 에서 컴파일이 실패하면 `npu_train.diagnose.first_failing_node` 로 처음 실패하는 노드를 찾고,
의미를 바꾸지 않는 변환을 `graph.PASSES` 에 넣는다. 모든 변환은 `graph.build` 가 CPU 재실행으로 결과가
같은지 확인한다.

## 컨테이너에서 컴파일할 때

공식 `furiosaai/furiosa-llm` 이미지에는 컴파일러가 호출하는 aarch64 크로스 툴체인이 없다(서빙은 미리 빌드한
artifact 만 쓴다). 없으면 첫 컴파일이 `RuntimeError: No such file or directory (os error 2)` 로 실패한다
(strace: `execve("/usr/bin/aarch64-linux-gnu-gcc") = ENOENT`, 이어서 `assert.h` 없음). 설치:

```bash
apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross libc6-dev
```

`npu-lora job` 과 `lmd-top --plan train` 이 만드는 Job 은 이 설치를 먼저 한다. 컴파일러는 또
`~/.cache/furiosa/compiler` 를 스스로 만들지 않으므로 백엔드가 미리 만든다.

## 런타임: 여러 스레드에서 장치 메모리 할당 (#40)

`torch.OutOfMemoryError: failed to allocate … bytes on DRAM_SHARED` 가 전체 파인튜닝 레이어 작업 스레드의 업로드에서
간헐적으로 났다(`--resident-layers` 실행, 3–13 step 뒤). 원인은 메모리 부족이 아니다.

- furiosa-torch 2026.2 는 장치 메모리 종류(DRAM / DRAM_SHARED)를 프로세스 전체에 하나뿐인 플래그로 둔다
  (`furiosa.native_torch.memory._set_dram_shared_mode` / `_set_memory_type`). 한 스레드에서 켜면 다른 스레드도 켜진 값을 본다.
- 컴파일된 그래프 호출은 매번 이 플래그를 잠깐 DRAM_SHARED 로 바꾼다(x·2 그래프 300번을 부르는 동안 다른 스레드가 플래그를 800만 번 읽어 1978번 DRAM_SHARED 를 봄).
- 그 순간 다른 스레드가 장치 텐서를 만들면 DRAM_SHARED 에 잡히는데, 칩 1개 카드에서 그 영역은 0 바이트라 바로 실패한다.
  보통 할당과 그래프 출력은 모두 DRAM(47.5 GiB 한 영역)에 잡힌다.
- 메인 스레드가 그래프를 도는 동안 작업 스레드가 올리는 시간이 길수록 잘 겹친다. 상주 실행은 메인 스레드가 상주 AdamW
  그래프를 돌리는 바로 그때 작업 스레드들이 업로드해서 몇 step 안에 걸렸고, 오프로드만 하는 실행은 덜 겹쳐서 운 좋게 피했다.

대응: `backends.DEVICE_ALLOC_LOCK` — 그래프 호출(`Serialized`)과 장치 할당(`FuriosaBackend.to_device` 의 `torch.empty`)이
같은 잠금을 잡는다. 복사(`copy_`)는 잠금 밖이라 업로드 속도는 같다(117 MB bf16: `t.to` 6.05 GB/s, `empty`+`copy_` 6.01 GB/s).
재현: `experiments/data/dram-shared/race_repro.py`(잠금 없이 그래프 6번 만에 OOM, 잠금으로 120 s·그래프 58 799번 동안 없음).
