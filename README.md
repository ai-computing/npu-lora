# npu-lora

**LoRA and full fine-tuning on inference-only NPUs.**

npu-lora trains Llama-family models on the FuriosaAI RNGD — an accelerator whose SDK ships only an
inference compiler and runtime. It turns the backward pass into ordinary compiled graphs, keeps the
optimizer on the host, and produces standard Hugging Face artifacts (PEFT adapters or full models)
that vLLM and llm-d serve as-is. The same code runs on CUDA and CPU for reference runs.

| | |
| --- | --- |
| **Hardware** | FuriosaAI RNGD (furiosa-torch 2026.2) · NVIDIA GPU (reference) · CPU (reference) |
| **Models** | Llama 2/3/3.1/3.2 architecture (GQA, RoPE incl. llama3 scaling, SwiGLU) |
| **Modes** | LoRA (q/k/v/o) · full fine-tuning of decoder layers with CPU offload |
| **Outputs** | PEFT adapter (`adapter_model.safetensors`) · full HF model · merged model |
| **Integrations** | vLLM runtime LoRA loading · llm-d · Kubernetes Jobs · [lmd-top](https://github.com/ai-computing/lmd-top) `--plan train` · Prometheus metrics |

## Results

Same data, seed and hyper-parameters on RNGD and A100 (Llama-3.1-8B-Instruct, LoRA r=16 on q·v,
400 steps; evaluation with vLLM on held-out data):

| | NSMC test acc. (n=1000) | MMLU acc. (n=400) |
| --- | ---: | ---: |
| Base model | 79.6% | 60.8% |
| **Trained on RNGD** | **87.5%** | **63.0%** |
| Trained on A100 | 87.6% | 63.0% |

- Gradients match a CPU reference: cosine ≥ 0.9997 (Llama-3.1-8B layers + 128k-vocab head).
- One RNGD card trains Llama-3.1-8B LoRA at ~0.7 s per 128-token step.
- Full fine-tuning is validated on Llama-3.2-1B.

See [docs/validation.md](docs/validation.md) and [`experiments/`](experiments) for the raw numbers.

## Installation

```bash
git clone https://github.com/ai-computing/npu-lora.git
cd npu-lora
pip install -e .
```

Requirements:

- Python ≥ 3.10, PyTorch ≥ 2.4, `transformers`, `safetensors`, `huggingface_hub`.
- **RNGD:** the Furiosa SDK with `furiosa-torch` (tested 2026.2.0; the `furiosaai/furiosa-llm`
  image contains a matching set) and the aarch64 cross toolchain that the compiler calls:

  ```bash
  apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross libc6-dev
  ```

- **Evaluation:** `vllm` (only for `npu_lora.bench`).

## Quick start

```bash
# 1. Verify that RNGD gradients match the CPU reference (first 2 layers).
npu-lora check --model meta-llama/Llama-3.1-8B-Instruct --backend furiosa:0 --layers 2

# 2. Train a LoRA adapter. Data is JSONL: {"text"} | {"prompt","response"} | {"messages"}.
npu-lora train --model meta-llama/Llama-3.1-8B-Instruct --data train.jsonl \
    --backend furiosa:0 --seq-len 128 --steps 400 --accum 4 --lr 2e-4 --out ./adapter

# 3. Serve it from a running vLLM (--enable-lora, VLLM_ALLOW_RUNTIME_LORA_UPDATING=True) and evaluate.
npu-lora serve --endpoint http://vllm:8000 --name my-adapter --path /models/adapter
npu-lora eval  --endpoint http://vllm:8000 --model my-adapter --data eval.jsonl
```

The first run compiles the training graphs part by part (Llama-8B: ~1.5 h on a 64-core host) and
caches them under `~/.cache/npu-lora`; later runs start immediately. `npu-lora compile` fills the
cache without opening a device, so you can compile on any host and copy the cache.

### Full fine-tuning

```bash
npu-lora train --full --model unsloth/Llama-3.2-1B-Instruct --data mix.jsonl \
    --backend furiosa:0 --seq-len 128 --steps 800 --accum 4 --lr 1e-5 --out ./model
```

Decoder-layer weights are trained; embeddings, `lm_head` and the final norm stay frozen. The output
is a complete Hugging Face model directory. Mix general instruction data into narrow tasks — training
on a single task collapses general ability on any device (see [docs/validation.md](docs/validation.md)).

### On Kubernetes

```bash
npu-lora job --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/train.jsonl \
    --name my-run --node npu-node-1 | kubectl apply -f -
```

The Job uses the official `furiosa-llm` image with the package mounted from a ConfigMap, requests one
`furiosa.ai/rngd` card, writes the result to the shared model store and exposes Prometheus metrics with
a ServiceMonitor. [lmd-top](https://github.com/ai-computing/lmd-top) builds the same Job with
`lmd-top --plan train --vendor furiosa --model … --set data=… [--set full=true]`.

## Python API

```python
from npu_lora import LoRATrainer, LoRAConfig, TrainConfig
from npu_lora import backends, export
from npu_lora.data import PackedDataset
from transformers import AutoTokenizer

trainer = LoRATrainer(
    "meta-llama/Llama-3.1-8B-Instruct",
    backends.make("furiosa:0"),            # or "cuda:0", "cpu"
    LoRAConfig(r=16, alpha=32, targets=("q", "v")),
    TrainConfig(seq_len=128, lr=2e-4, steps=400, accum=4),
)
trainer.setup()                             # load weights, compile (or load) graphs

tok = AutoTokenizer.from_pretrained(trainer.path)
data = PackedDataset("train.jsonl", tok, seq_len=128)
for batch in data.batches(accum=4, steps=400):
    print(trainer.train_step(batch))        # {"step", "loss", "grad_norm", "tokens_per_s", ...}

export.save_peft(trainer, "./adapter")      # export.save_full(...) with TrainConfig(full=True)
```

## How it works

1. **Weights are graph inputs**, so one compiled decoder-layer graph serves every layer.
2. **Backward is a traced graph** (`make_fx` over `torch.func.vjp`) that the inference compiler runs
   like any forward graph, after semantics-preserving rewrites for patterns it cannot handle.
3. **Graphs are compiled in parts of two matmuls**; large fused graphs exhaust the compiler.
4. **The optimizer runs on the host.** Full fine-tuning overlaps device→host gradient transfer,
   host AdamW and host→device upload with the backward pass and the next forward pass.

Details: [docs/design.md](docs/design.md). Compiler issues and their workarounds:
[docs/compiler-workarounds.md](docs/compiler-workarounds.md).

## Commands

| Command | Purpose |
| --- | --- |
| `npu-lora train` | LoRA or `--full` fine-tuning |
| `npu-lora check` | Compare device gradients with the CPU reference |
| `npu-lora compile` | Compile and cache training graphs without a device |
| `npu-lora merge` | Merge an adapter into the base weights (for NPU serving, which recompiles) |
| `npu-lora serve` | Load an adapter into a running vLLM |
| `npu-lora eval` | Keyword-accuracy evaluation against an OpenAI-compatible endpoint |
| `npu-lora job` | Print a Kubernetes training Job |
| `python -m npu_lora.bench` | vLLM offline benchmark of base / adapters / full models |
| `python -m npu_lora.compare` | Compare two training loss curves |
| `python -m npu_lora.hwprobe` | Measure host↔device bandwidth and transfer/compute overlap |

Frequently used training options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | `furiosa:0` | `furiosa:N`, `furiosa:auto` (card assigned by Kubernetes), `cuda:N`, `cpu` |
| `--seq-len` | 512 | Sequence length; fixed at compile time (a new length means a new compile) |
| `--rank`, `--alpha`, `--targets` | 16, 32, `q,v` | LoRA settings (`q,k,v,o`) |
| `--accum` | 1 | Gradient accumulation (micro-batch = one packed sequence) |
| `--full` | off | Full fine-tuning of decoder-layer weights |
| `--max-mm` | 2 on RNGD | Matmuls per compiled graph part |
| `--metrics-port` | — | Serve Prometheus metrics (`npu_lora_loss`, `npu_lora_tokens_per_second`, …) |

Training progress is also written as JSON lines to stdout and `--log-file`.

## Development

```bash
python tests/test_cpu.py        # CPU regression tests (no NPU needed)
```

The tests pin the issues that produced silently wrong results during development: graph rewrites and
partitioning must not change values, duplicate sample tensors must not merge inputs, the vocab-chunked
loss and RoPE scaling must match Hugging Face, and the full fine-tuning pipeline must equal a
synchronous update.

When you bring up a new model or SDK version:

1. `npu-lora check` must pass (device vs CPU gradients).
2. If compilation fails, `npu_lora.diagnose.first_failing_node` locates the first failing op; add a
   semantics-preserving rewrite to `graph.PASSES`.
3. If results are wrong but compilation succeeds, `npu_lora.diagnose.compare_parts` compares every
   compiled part with the CPU on the same inputs.

`scripts/vendor-into-lmd-top.sh` copies the package into lmd-top, which embeds it in the Jobs it
generates.

## Limitations

- Llama architecture only. Other architectures need their layer written as a pure function
  (see `npu_lora/llama.py`).
- One packed sequence per micro-batch; use `--accum` for larger batches.
- First compilation is slow (hours for an 8B model) and is per sequence length.
- furiosa-torch opens devices only from privileged containers.
- Rebellions ATOM is not supported yet; the same approach applies (`rebel.compile_from_torch`).

## Project status

Research code from the ETRI llm-d cluster. Current state and open questions are tracked in
[STATUS.md](STATUS.md); measurements in [`experiments/`](experiments).

## ai-rm

[ai-rm](https://github.com/ai-computing/ai-rm)(AI 반도체 통합 자원관리)의 구성요소다. 설정은 [`ai-rm.yaml`](ai-rm.yaml).

- **계층·역할**: `l3-serving`, kind `workload`. 추론 전용 AOT 컴파일 NPU인 FuriosaAI RNGD에서 Llama 계열
  모델을 LoRA 또는 전체 파인튜닝하는 학습 워크로드다. 결과는 HF PEFT 어댑터나 HF 전체 모델이다.
- **k8s + llm-d 연동**: 학습은 쿠버네티스 Job으로 돈다. `npu-lora job … | kubectl apply -f -`가
  ConfigMap(패키지), Job(`furiosa.ai/rngd` 1장, `furiosaai/furiosa-llm` 이미지, privileged), Service,
  ServiceMonitor를 만들고 결과를 공유 모델 스토어 PVC(`model-store`)에 쓴다. 산출 어댑터는 vLLM
  `--enable-lora`와 llm-d LoRA 라우팅이 그대로 읽고, `npu-lora serve`로 실행 중인 vLLM에 재시작 없이 올린다.
  lmd-top은 `lmd-top --plan train`으로 같은 Job을 만든다.
  아직 없는 것: 레포에 커밋된 정적 매니페스트(kustomize 디렉터리)와 전용 컨테이너 이미지. 매니페스트는
  CLI가 그때그때 생성한다.
- **설치·테스트**: `python -m pip install -e .`, `python tests/test_cpu.py`(CPU 회귀 테스트, NPU 불필요,
  PyTorch 필요). RNGD 학습에는 Furiosa SDK(`furiosa-torch`)가 필요하다.
- **인터페이스**: 제공 `Workload`(학습 Job), `Metrics`(`--metrics-port` Prometheus 지표 `npu_lora_loss`,
  `npu_lora_tokens_per_second` 등과 ServiceMonitor). 다른 ai-rm 구성요소의 인터페이스는 아직 쓰지 않는다
  (장치는 `furiosa.ai/rngd` 리소스와 `NPU_LORA_RNGD`/`NPU_LORA_RNGD_FILE`로 직접 고른다).
- **KPI**: 이종자원수 — 추론 전용 NPU를 학습 자원으로 쓴다. RNGD 학습 품질이 A100과 같음을
  [docs/validation.md](docs/validation.md)에서 확인했다.
- **`cluster.local.env`**: 라이브러리와 CLI는 사이트 값을 쓰지 않는다(`--endpoint` 등은 인자로 받는다).
  데모 데이터 생성기 `examples/cluster_facts.py`만 아래 값을 읽는다. `cluster.example.env`를
  `cluster.local.env`로 복사해 채운다.

  | 변수 | 뜻 |
  | --- | --- |
  | `MODEL_STORE_NAS_HOST` | 모델 스토어 NAS 주소 |
  | `LLMD_GATEWAY_HOST` | llm-d 게이트웨이 주소 |
  | `PROMETHEUS_HOST`, `PROMETHEUS_PORT` | 클러스터 Prometheus 주소와 포트 |
