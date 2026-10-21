<div align="center">

<h1>npu-train</h1>

<h3>LoRA and full fine-tuning on inference-only NPUs</h3>

<p>
  <strong>English</strong> · <a href="README.ko.md">한국어</a>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L3%20execute%20%26%20serve-34d399" alt="ai-rm layer: L3 execute and serve"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.4 or newer"></a>
  <img src="https://img.shields.io/badge/hardware-FuriosaAI%20RNGD-f97316" alt="Hardware: FuriosaAI RNGD">
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-prototype-eab308" alt="Maturity: prototype"></a>
</p>

<p>
  <a href="#-highlights"><b>Highlights</b></a> ·
  <a href="#-results"><b>Results</b></a> ·
  <a href="#-quick-start"><b>Quick start</b></a> ·
  <a href="#-ai-rm-integration"><b>ai-rm integration</b></a> ·
  <a href="#-architecture"><b>How it works</b></a> ·
  <a href="#-documentation"><b>Docs</b></a>
</p>

</div>

---

Formerly the `npu-lora` repository; the Python package, CLI and Helm chart keep the name `npu-lora`.

**npu-train** (Python package and CLI `npu-lora`) trains Llama-family models on the FuriosaAI RNGD — an accelerator whose SDK ships only an
inference compiler and runtime. It turns the backward pass into ordinary compiled graphs, keeps the
optimizer on the host, and produces standard Hugging Face artifacts (PEFT adapters or full models)
that vLLM and llm-d serve as-is. The same code runs on CUDA and CPU for reference runs.

In [**ai-rm**](https://github.com/ai-computing/ai-rm) it is the L3 training workload: it registers the
`npu-lora` workload runner (`airm.workloads:npu-lora`), reports training throughput as `airm_tok_s`, and
ships a Helm chart for training Jobs. Its KPI contribution is heterogeneous resource use: an inference-only
NPU used as a training resource.

## ✨ Highlights

<table>
<tr>
<td width="33%" valign="top">
<b>Training on an inference compiler</b><br>
Weights are graph inputs and the backward pass is a traced graph, so the RNGD inference compiler runs training like any forward graph.
</td>
<td width="33%" valign="top">
<b>Same quality as A100</b><br>
Llama-3.1-8B LoRA trained on RNGD matches the same run on A100 (NSMC 87.5% vs 87.6%, MMLU 63.0% both), and gradients match a CPU reference.
</td>
<td width="33%" valign="top">
<b>Standard outputs, cluster ready</b><br>
PEFT adapters or full HF models that vLLM <code>--enable-lora</code> and llm-d load directly; Kubernetes Jobs, a Helm chart and Prometheus metrics.
</td>
</tr>
</table>

| | |
| --- | --- |
| **Hardware** | FuriosaAI RNGD (furiosa-torch 2026.2) · NVIDIA GPU (reference) · CPU (reference) |
| **Models** | Llama 2/3/3.1/3.2 architecture (GQA, RoPE incl. llama3 scaling, SwiGLU) |
| **Modes** | LoRA (q/k/v/o) · full fine-tuning of decoder layers with CPU offload |
| **Outputs** | PEFT adapter (`adapter_model.safetensors`) · full HF model · merged model |
| **Integrations** | vLLM runtime LoRA loading · llm-d · Kubernetes Jobs · [lmd-top](https://github.com/ai-computing/lmd-top) `--plan train` · Prometheus metrics |

## 📊 Results

Same data, seed and hyper-parameters on RNGD and A100 (Llama-3.1-8B-Instruct, LoRA r=16 on q·v,
400 steps; evaluation with vLLM on held-out data):

| | NSMC test acc. (n=1000) | MMLU acc. (n=400) |
| --- | ---: | ---: |
| Base model | 79.6% | 60.8% |
| **Trained on RNGD** | **87.5%** | **63.0%** |
| Trained on A100 | 87.6% | 63.0% |

- Gradients match a CPU reference: cosine ≥ 0.9997 (Llama-3.1-8B layers + 128k-vocab head).
- One RNGD card trains Llama-3.1-8B LoRA at ~0.7 s per 128-token step.
- Llama-3.1-8B full fine-tuning on one RNGD card (CPU offload) matches the same run on A100: NSMC 84.6% vs 84.2%,
  MMLU 52.75% vs 50.25% ([record](experiments/2026-10-21-full8b-quality.md)); LoRA keeps more general ability (MMLU 63%).

See [docs/validation.md](docs/validation.md) and [`experiments/`](experiments) for the raw numbers.

Training energy on one card, same node and recipe (card power from Prometheus; [record](experiments/2026-10-21-energy-rngd-vs-a100.md)):

| Llama-3.1-8B | RNGD tok/s | RNGD tok/J | A100 tok/s | A100 tok/J |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning, S=512, accum 4, CPU offload | 332 | **4.32** | 492 | 3.08 |
| LoRA r=16 q·v, S=128 (npu-lora on both) | 49 | 1.05 | 366 | 2.41 |
| LoRA, A100 with plain PyTorch autograd | — | — | 1,683 | 6.39 |

Counting the host power of the offload path, full fine-tuning is 1.92 (RNGD) vs 1.55 (A100) tok/J. RNGD LoRA on
current `main` is 4× slower than the 2026-10-16 run (more graph parts per layer), so its tok/J is not RNGD's limit.

> [!NOTE]
> **Prototype.** LoRA and full fine-tuning quality are validated on Llama-3.1-8B (full fine-tuning on one card with
> CPU offload) and Llama-3.2-1B. Current state and open questions: [STATUS.md](STATUS.md).

## 🚀 Quick start

```bash
git clone https://github.com/ai-computing/npu-train.git
cd npu-train
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

<details>
<summary><b>Full fine-tuning</b></summary>
<br>

```bash
npu-lora train --full --model unsloth/Llama-3.2-1B-Instruct --data mix.jsonl \
    --backend furiosa:0 --seq-len 128 --steps 800 --accum 4 --lr 1e-5 --out ./model
```

Decoder-layer weights are trained; embeddings, `lm_head` and the final norm stay frozen. The output
is a complete Hugging Face model directory. Mix general instruction data into narrow tasks — training
on a single task collapses general ability on any device (see [docs/validation.md](docs/validation.md)).
</details>

<details>
<summary><b>On Kubernetes</b></summary>
<br>

```bash
npu-lora job --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/train.jsonl \
    --name my-run --node <node> | kubectl apply -f -
```

The Job uses the official `furiosa-llm` image with the package mounted from a ConfigMap, requests one
`furiosa.ai/rngd` card, writes the result to the shared model store and exposes Prometheus metrics with
a ServiceMonitor. [lmd-top](https://github.com/ai-computing/lmd-top) builds the same Job with
`lmd-top --plan train --vendor furiosa --model … --set data=… [--set full=true]`.
</details>

<details>
<summary><b>Python API</b></summary>
<br>

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
</details>

<details>
<summary><b>Commands and options</b></summary>
<br>

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
| `--metrics-port` | — | Serve Prometheus metrics (`airm_tok_s`, `npu_lora_loss`, `npu_lora_step_seconds`, …) |

Training progress is also written as JSON lines to stdout and `--log-file`.
</details>

## 🔌 ai-rm integration

npu-lora is an ai-rm component (layer `l3-serving`, kind `workload`, maturity `prototype`); configuration
summary in [`ai-rm.yaml`](ai-rm.yaml). It trains Llama-family models with LoRA or full fine-tuning on the
FuriosaAI RNGD, an inference-only ahead-of-time-compiled NPU, and produces HF PEFT adapters or full HF models.

| Entry-point group | Name | Object |
| --- | --- | --- |
| `airm.workloads` | `npu-lora` | `npu_lora.airm:train` |

```python
from rm_core import Workload, registry
run = registry.load("airm.workloads", "npu-lora", backend="rngd")    # constructor kwargs = config defaults
res = run(Workload(kind="train", model="meta-llama/Llama-3.1-8B-Instruct"),
          data="/mnt/store/datasets/nsmc.jsonl", out="/mnt/store/adapters/nsmc", dataset="nsmc", steps=400)
res["measurements"]      # [Measurement("tok_s", …, tags={"workload": "train", "dataset": "nsmc", …})]
```

The runner wraps the existing `npu-lora train`; the training path is the same.

**Install and test:** `python -m pip install -e .`, `python tests/test_cpu.py` (CPU regression tests, no NPU,
needs PyTorch), `python tests/test_airm.py` (ai-rm runner, CPU), `python -m rm_core.conformance`. Training on
RNGD needs the Furiosa SDK (`furiosa-torch`).

<details>
<summary><b>Kubernetes, llm-d and the Helm chart</b></summary>
<br>

Training runs as a Kubernetes Job. `npu-lora job … | kubectl apply -f -` creates a ConfigMap (the package),
a Job (one `furiosa.ai/rngd` card, `furiosaai/furiosa-llm` image, privileged), a Service and a ServiceMonitor,
and writes the result to the shared model-store PVC (`model-store`). The adapters it produces are read as-is
by vLLM `--enable-lora` and llm-d LoRA routing, and `npu-lora serve` loads them into a running vLLM without a
restart. lmd-top builds the same Job with `lmd-top --plan train`.

The ai-rm umbrella chart uses the Helm chart [`deploy/helm/npu-lora`](deploy/helm/npu-lora). Depending on
`train.backend` (`rngd|cuda|cpu`) it selects nodes by the `airm.ai-computing.io/<vendor>.<model>` label
(rngd: `airm.ai-computing.io/furiosa.RNGD`; otherwise set `nodeAffinity`), and keeps data, checkpoints and
cache on the model-store PVC (`store.pvc`). If the image does not contain npu-lora, it is installed at start
with `install.pip`. The metrics port carries the `airm.ai-computing.io/scrape: "true"` annotation.

```bash
helm template deploy/helm/npu-lora --set train.model=… --set train.data=…
```
</details>

<details>
<summary><b>Workload runner configuration</b></summary>
<br>

`Workload` accepts only `kind="train"` and `precision="bf16"` (anything else raises `ContractError`).

| Key | Meaning (default) |
| --- | --- |
| `data`, `out` | Required. Training JSONL, output directory |
| `backend` | `rngd` · `rngd:N` · `cuda[:N]` · `cpu` (`rngd`) |
| `dataset` | Value of the metric tag `dataset` (none) |
| `rank`, `alpha`, `targets`, `seq_len`, `steps`, `accum`, `lr`, `warmup`, `grad_clip`, `seed`, `full`, `save_every`, `merge_out`, `run_name`, `log_file`, `metrics_port`, `cache_dir`, `tactic`, `max_shapes`, `max_mm` | The `npu-lora train` option of the same name |
| `kv_transport` | Training has no KV path. Leave it empty (setting it raises `ContractError`) |
</details>

<details>
<summary><b>Metrics</b></summary>
<br>

Using rm-core `METRICS` names: training throughput `tok_s` (Prometheus `airm_tok_s`, tags
`component=npu-lora`, `workload=train`, `model`, `vendor`, `run_id`; also in the runner's return value).
Accuracy `accuracy_pct` is in the `measurements` of the `npu_lora.bench` result JSON (tag
`dataset=nsmc|mmlu`, `run_id` = variant name). Quantities not in rm-core are exported as `npu_lora_*`
(`npu_lora_loss`, `npu_lora_grad_norm`, `npu_lora_step_seconds`, …).

**Deprecated**: the old throughput series `npu_lora_tokens_per_second` (label `run`) is exported with the
same value as `airm_tok_s` only until protocol 0.2 (PROTOCOL §6 transition rule). Consumers read `airm_tok_s`
first and fall back to the old name only when it is missing. The common name for step time is the rm-core
candidate `step_ms`. `power_w` is not exported yet. The device is chosen through the `furiosa.ai/rngd`
resource and `NPU_LORA_RNGD` / `NPU_LORA_RNGD_FILE`.
</details>

<details>
<summary><b>Site configuration (<code>cluster.local.env</code>)</b></summary>
<br>

The library and the CLI use no site values (`--endpoint` and similar are arguments). Only the demo data
generator `examples/cluster_facts.py` reads the values below. Copy [`cluster.example.env`](cluster.example.env)
to `cluster.local.env` and fill it in.

| Variable | Meaning |
| --- | --- |
| `MODEL_STORE_NAS_HOST` | Model-store NAS address |
| `LLMD_GATEWAY_HOST` | llm-d gateway address |
| `PROMETHEUS_HOST`, `PROMETHEUS_PORT` | Cluster Prometheus address and port |
| `GPU_NODE`, `NPU_NODE`, `GB10_NODE_1`, `GB10_NODE_2` | Node names (optional; neutral aliases when unset) |
</details>

## 🏗 Architecture

1. **Weights are graph inputs**, so one compiled decoder-layer graph serves every layer.
2. **Backward is a traced graph** (`make_fx` over `torch.func.vjp`) that the inference compiler runs
   like any forward graph, after semantics-preserving rewrites for patterns it cannot handle.
3. **Graphs are compiled in parts of two matmuls**; large fused graphs exhaust the compiler.
4. **The optimizer runs on the host.** Full fine-tuning overlaps device→host gradient transfer,
   host AdamW and host→device upload with the backward pass and the next forward pass.

Details: [docs/design.md](docs/design.md). Compiler issues and their workarounds:
[docs/compiler-workarounds.md](docs/compiler-workarounds.md).

<details>
<summary><b>Development</b></summary>
<br>

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
</details>

### Limitations

- Llama architecture only. Other architectures need their layer written as a pure function
  (see `npu_lora/llama.py`).
- One packed sequence per micro-batch; use `--accum` for larger batches.
- First compilation is slow (hours for an 8B model) and is per sequence length.
- furiosa-torch opens devices only from privileged containers.
- Rebellions ATOM is not supported yet; the same approach applies (`rebel.compile_from_torch`).

## 📚 Documentation

| Document | Content |
| --- | --- |
| [docs/design.md](docs/design.md) | Why training works on an inference-only NPU, vocabulary-chunked loss, CPU-offload pipeline, Job layout |
| [docs/validation.md](docs/validation.md) | PoC, RNGD vs A100 quality, full fine-tuning results (Korean) |
| [docs/compiler-workarounds.md](docs/compiler-workarounds.md) | furiosa-torch 2026.2 compiler issues and workarounds (Korean) |
| [`experiments/`](experiments) | Experiment records and raw data (`experiments/data/`) |
| [STATUS.md](STATUS.md) | Current state, findings and open questions (Korean) |

Research code from the ETRI llm-d cluster.
