# 컴파일러 우회 (furiosa-torch 2026.2)

추론용 AOT 컴파일러에 역전파 그래프를 넣으면서 만난 문제와 대응이다. 모든 대응은 `npu_lora/graph.py` 의
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
| `npu-lora check` 가 엉뚱한 차이 | furiosa 를 불러오면 `torch.manual_seed` 가 장치 쪽에서 경고만 내고 끝나 두 trainer 의 A 초기값이 달라짐 | 어댑터 초기화에 전용 `torch.Generator` |

값이 틀리는 문제는 컴파일 오류가 나지 않는다. `npu_lora.diagnose.compare_parts` 로 조각마다 CPU 와 장치 출력을
같은 입력으로 비교한다. 학습 루프의 컴파일 샘플은 0/상수라서 이런 문제를 잡지 못하니, 새 모델은 반드시
`npu-lora check` 로 기울기를 비교한다.

새 모델·새 SDK 에서 컴파일이 실패하면 `npu_lora.diagnose.first_failing_node` 로 처음 실패하는 노드를 찾고,
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
