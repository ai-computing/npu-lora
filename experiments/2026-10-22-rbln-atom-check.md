# RBLN (ATOM): the SiLU-backward compile failure, ATOM rounding, and a passing `npu-lora check`

- **Date**: 2026-10-22
- **Task**: #15 (draft PR #20)
- **Host / device**: npu-node-1, one RBLN-CA22 (ATOM) card used directly (no Kubernetes), a card with no process on it
  per `rbln-stat`. rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, torch 2.11.0+cpu, transformers 5.8.1.
- **Commit**: `feat/15-rbln-backend` (PR #20 rebased on `main` 2b23c6a)

## Question

PR #20 compiled 50 of 51 graph parts on ATOM, but one backward part crashed the compiler, and `npu-lora check`
(2 layers) was 16 % off the CPU. Two questions:
- Which op breaks the compiler, and can npu-train avoid it?
- Where does the gradient error come from, and which parts must run where for the check to pass?

## SDK

Newest installed: rebel-compiler 0.11.0 (user site; 0.10.2 and 0.11.0 under /usr/local), optimum-rbln 0.11.0.post1,
vllm-rbln 0.11.0.

A newer SDK exists but could not be installed:
- Public PyPI has optimum-rbln 0.12.0, vllm-rbln 0.12.0 and torch-rbln 0.12.0, all uploaded 2026-10-16.
- They require torch 2.13.0+cpu, transformers 5.15.1 and `rebel-compiler>=0.12.0,<0.12.1`.
- rebel-compiler is only on the vendor index `pypi.rbln.ai`. It answers 401, and the node has no credentials for it.
- Nothing was upgraded. Every result below is rebel-compiler 0.11.0.

## Setup

- Model: `unsloth/Llama-3.2-1B-Instruct`. `check` used its first 2 layers, S=128, LoRA r16 on q·v, B ~ N(0, 0.01).
- The `check` inputs were random ids. Seed 1 is what `check` uses; seeds 2–4 vary the ids and B.
- Graph parts: `--max-mm 2`, 59 parts (layer forward 15, layer backward 39, head 5).
- For the diagnostics, every ATOM part call was also run eagerly on the CPU with the same inputs.
- An fp32 autograd reference (`LoRATrainer.reference_grads`, now reported by `check`) used the same bf16 weights
  and adapters, with no bf16 rounding between ops.

## Results

### 1. The compile failure: `neg` + `add.Scalar`

The failing part was the SiLU backward: `sigmoid`, `neg`, `add.Scalar`, `mul`, `mm`, at [128, 8192] @ [8192, 2048]
bf16. The compiler process segfaulted (exit −11).

Variants of the part were compiled in separate processes:
- The same math re-traced with `make_fx` compiled. That trace writes `1 − s` as `rsub`.
- `exp`/reciprocal, `tanh` and fp32 forms of sigmoid all compiled, and so did the saved graph after only
  `neg + add.Scalar(1)` was replaced by `rsub.Scalar(1)`.
- Fixing the metadata, re-tracing the saved graph or renaming its inputs did not help.

The minimal repro is `data/rbln-atom/repro_neg_add.py`. These are hand-built aten graphs, compiled only:

| graph | bf16 | fp32 |
|---|---|---|
| `add.Scalar(neg(x), 1.0)` | DEVICE_GRAPH_CONVERSION | DEVICE_GRAPH_CONVERSION |
| `add.Tensor(neg(x), 1.0)` | DEVICE_GRAPH_CONVERSION | DEVICE_GRAPH_CONVERSION |
| `rsub.Scalar(x, 1.0)`, `neg(x)`, `add.Scalar(x, 1.0)`, `add.Scalar(mul.Scalar(x, −1), 1.0)` | ok | ok |
| `mm(x * add.Scalar(neg(x), 1.0), w)`, [128, 256] @ [256, 64] | DEVICE_GRAPH_CONVERSION | DEVICE_GRAPH_CONVERSION |
| the real part, [128, 8192] @ [8192, 2048] | segfault | — |

The failure does not depend on the shapes [4, 64] and [128, 8192].

npu-train's `graph.scalarize_constants`, a furiosa workaround, writes `1 − sigmoid(x)` exactly this way.

Fix: `graph.fold_neg_add` rewrites `add(neg(x), c)` → `rsub(x, c)`. Only the RBLN backend applies it. The `.rbln`
cache path now hashes the rewritten graph, so old `.host` markers are not reused. All 59 parts now compile. The
SiLU-backward part runs on ATOM within 0.7 % of the CPU.

### 2. ATOM rounding

Single ops on random bf16 inputs were compared with a float64 truth (`data/rbln-atom/repro_rounding.py`):
- `bias` = mean(err · sign(truth)) / mean(|err|). It is 0 for unbiased rounding and +1 if every error makes |out|
  larger.
- `≠ CPU` is the fraction of elements that differ from torch's round-to-nearest-even result.

| op | ATOM rel. err | ATOM bias | CPU rel. err | CPU bias | ≠ CPU |
|---|---:|---:|---:|---:|---:|
| bf16 `x + y` | 1.86e-3 | +0.63 | 1.72e-3 | −0.01 | 14 % |
| bf16 `x * y` | 1.82e-3 | +0.48 | 1.65e-3 | −0.03 | 14 % |
| bf16 `x @ wᵀ` (K=2048) | 3.59e-3 | +0.77 | 1.66e-3 | 0.00 | 59 % |
| `x.float() @ w.float()ᵀ` (fp32 out) | 2.70e-3 | +0.70 | 1.1e-7 | 0.00 | 100 % |
| fp32 → bf16 cast of an fp32 expression | 2.38e-3 | +0.88 | 1.66e-3 | −0.01 | 31 % |
| fp32 `x + 0.0078 y` | 4.2e-4 | +0.17 | 7e-10 | +0.16 | 79 % |

- ATOM rounds away from zero much more often than round-to-nearest-even. The error systematically enlarges
  magnitudes.
- An fp32 output is no more precise than bf16. A graph that asks for fp32 logits gets bf16-level logits, about 0.1
  absolute at |logit| ≈ 30.

### 3. Where the `check` error comes from

Per part, with the same inputs, all on ATOM, seed 1:
- Forward parts are within 0.6 % of the CPU and backward parts within 0.75 %.
- `ce-stats` is within 0.8 %. `ce-grad` is up to 4.3 % (mean 2.5 % over its 32 calls): `exp(logits − lse)` turns
  the logit error into a relative error of p.

End to end, `check`'s gradient comparison, worst over the 8 LoRA tensors:

| host parts (rest on ATOM) | seed | vs CPU | min cos | ATOM vs fp32 | CPU vs fp32 |
|---|---:|---:|---:|---:|---:|
| none (all on ATOM) | 1 | **13.4 %** | 0.9911 | 4.8 % | 9.1 % |
| none | 2 | 10.1 % | 0.9957 | 8.9 % | 3.0 % |
| none | 3 | 2.2 % | 0.9998 | 2.4 % | 1.7 % |
| none | 4 | 3.1 % | 0.9996 | 2.4 % | 1.9 % |
| `fwd` (layer forward) | 1 | **1.4 %** | 0.99991 | 9.3 % | 9.1 % |
| `fwd` | 2 | 4.9 % | 0.9988 | 5.1 % | 3.0 % |
| `fwd` | 3 | 1.7 % | 0.9999 | 2.2 % | 1.7 % |
| `fwd` | 4 | 2.7 % | 0.9996 | 4.0 % | 1.9 % |
| head `ce-grad` only | 1 | 14.5 % | 0.9902 | | |
| head `rms`, `ce-stats`, `ce-grad` | 1 | 13.4 % | 0.9910 | | |
| `fwd` + `ce-grad` | 1 / 2 / 3 / 4 | 4.0 / 5.9 / 3.9 / 4.8 % | ≥ 0.9987 | | |
| fwd parts 0, 3–8 (rms1, q/k/v projections, q RoPE) | 2 | 4.5 % | 0.9990 | 3.2 % | 3.0 % |
| fwd parts 10, 12 (q·kᵀ, softmax, att·v, o) | 2 | 8.2 % | 0.9970 | 6.2 % | 3.0 % |
| fwd parts 13, 14 (residual, rms2, MLP) | 2 | 7.7 % | 0.9976 | 6.8 % | 3.0 % |

The loss is within 0.01–0.18 % of the CPU in every configuration.

- **The error comes from the forward activations that leave a layer.** They drift from the rounding above, and the
  head's softmax amplifies the drift. With the forward on the host, the gradients follow the CPU, even though the
  backward recomputes each layer's forward on ATOM.
- Most of the forward's effect comes from the q/k/v projection parts. Through q·kᵀ (logits up to |226| before
  scaling) they move the attention.
- The CPU bf16 reference is not exact either. Seed 1, where `check` failed, is a draw on which the CPU is 9.1 % off
  fp32 and all-ATOM 4.8 %. Over seeds 2–4, all-ATOM is 1.2–3× further from fp32 than the CPU.
- The head should stay on one side. With `ce-grad` on the host but `lse` from ATOM's `ce-stats`, the result is worse
  than with both on ATOM, because the two ATOM parts' logit errors cancel in `exp(logits − lse)`.

### 4. `npu-lora check` (CLI, seed 1)

| placement | loss rel. | worst grad vs CPU | min cos | pass |
|---|---:|---:|---:|---|
| `--host-parts none` (all 59 parts on ATOM) | 1.76e-3 | 13.36 % | 0.99114 | no |
| default `--host-parts fwd` (44 parts on ATOM, 15 on the host) | 1.84e-3 | **1.39 %** | 0.99991 | **yes** |

Raw: `data/rbln-atom/check-default.json`, `check-all-atom.json`, `summary.json`.

### 5. LoRA runs (full 16 layers)

Recipe: r16 q·v, S=128, accum 1, seed 0. The CPU run is the same code with `--backend cpu`.

| data, steps, lr | run | loss step 1 → 30/60 | window mean first → last 10 | median s/step (ATOM parts / host parts) |
|---|---|---|---|---|
| code-name set (64 ex.), 30, 5e-4 | ATOM default | 3.85 → 0.54 | 1.76 → 0.56 | 1.17 (0.88 / 0.13) |
| | CPU | 3.81 → 0.51 | 1.72 → 0.52 | 0.65 |
| NSMC train, 60, 2e-4 | ATOM default | 0.47 → 0.06 | 0.470 → 0.094 | 1.39 (1.09 / 0.14) |
| | ATOM all parts | 0.46 → 0.06 | 0.479 → 0.128 | 1.59 (1.43 / 0) |
| | CPU | 0.44 → 0.05 | 0.448 → 0.093 | 0.79 |

- Both ATOM curves follow the CPU curve step by step (`npu_lora.compare --window 10`). The median window difference
  is 4.9 % for the default and 6.9 % for all parts on ATOM.
- The reported ATOM loss sits about 0.035 nats above the CPU on confident tokens: the floor is 0.02–0.04 on ATOM vs
  0.001–0.005 on the CPU. The vocabulary statistics (lse) come from ATOM, whose logits are inflated by the rounding
  above, while the target logit is computed on the host (#43). The gradient uses the ATOM `lse` and ATOM logits
  together, so they stay consistent.
- Step time is not a result here:
  - The RBLN runtime takes host tensors and uploads every input per call, including the layer weights.
  - The CPU is faster than ATOM at this size.
  - The forward on the host costs about 0.14 s of a 1.4 s step.

Raw: `data/rbln-atom/{codename,nsmc}-*.jsonl`.

## Command

```bash
M=unsloth/Llama-3.2-1B-Instruct
python3 -m npu_lora.cli check --model $M --backend rbln:0 --layers 2 --seq-len 128 --max-mm 2            # default: fwd on host
python3 -m npu_lora.cli check --model $M --backend rbln:0 --layers 2 --seq-len 128 --max-mm 2 --host-parts none
python3 -m npu_lora.cli train --model $M --backend rbln:0 --max-mm 2 --data nsmc_train.jsonl --steps 60 --seq-len 128 \
  --lr 2e-4 --warmup 5 --seed 0 --rank 16 --alpha 32 --targets q,v --out adapter --log-file nsmc-atom-default.jsonl
python3 experiments/data/rbln-atom/repro_neg_add.py
python3 experiments/data/rbln-atom/repro_rounding.py 0
```

## Conclusion

- **Compiler bug**: rebel-compiler 0.11 cannot compile `add(neg(x), c)` (DEVICE_GRAPH_CONVERSION). With a large
  matmul in the same graph it segfaults. npu-train avoids it with `rsub`, and every graph part compiles.
- **Numerics**: ATOM's rounding is biased away from zero, and fp32 outputs carry bf16-level error.
  - With every part on ATOM, `check` is 2–13 % off the CPU depending on the inputs, and fails on the default inputs
    (13.4 %).
  - The default RBLN placement (layer forward on the host; layer backward and head on ATOM) passes: 1.39 %, cos
    0.99991.
  - Over seeds 1–4 this placement is at 1.4–4.9 % of the CPU.
- **Training**: LoRA training on ATOM reduces the loss and tracks the CPU curve.
- Not done:
  - quality against A100 / RNGD, and performance (the per-call weight upload);
  - the 0.12 SDK, which needs credentials for the vendor index.

## Vendor bug-report drafts (for the lead to send)

**1. rebel-compiler 0.11.0: `aten.add.Scalar(aten.neg(x), c)` fails DEVICE_GRAPH_CONVERSION; segfault with a
following matmul**

- Environment: RBLN-CA22, rebel-compiler 0.11.0, torch 2.11.0+cpu, Python 3.10, `rebel.compile_from_torch` on a
  `torch.fx.GraphModule`.
- Repro: `repro_neg_add.py` (attached; it builds the graphs from aten ops with torch.fx).
  - `add.Scalar(neg(x), 1.0)` on [4, 64] or [128, 8192], bf16 or fp32 → `RBLNCompileError: Graph Generation:
    [DEVICE_GRAPH_CONVERSION]`.
  - `rsub.Scalar(x, 1.0)` and each op alone compile.
  - Inside a SiLU-backward graph `mm(g * s * (add.Scalar(neg(s), 1) * x + 1), w)` with [128, 8192] @ [8192, 2048]
    bf16, the compiler process segfaults.
- Expected: compiles like `rsub`, the same value.
- Impact: any functionalized graph that writes `c − y` as `neg` + `add` (e.g. after constant folding).

**2. ATOM (RBLN-CA22, rebel-compiler 0.11.0): rounding biased away from zero; fp32 outputs have bf16-level error**

- Repro: `repro_rounding.py <device>` (attached). Random bf16 inputs; compare with float64 truth and torch CPU.
  - bf16 `x + y`: 14 % of elements differ from round-to-nearest-even. Bias +0.63, where bias is mean(err · sign(truth))
    / mean(|err|): 0 for unbiased rounding, +1 if every error enlarges |out|.
  - bf16 matmul (K=2048): 59 % differ, relative error 3.6e-3 vs 1.7e-3, bias +0.77.
  - A matmul with fp32 output (`x.float() @ w.float().t()`) has relative error 2.7e-3, the same as bf16.
- Questions:
  - Is round-to-nearest-even available, or is this rounding mode by design?
  - Is fp32 accumulation and output supported for matmul and elementwise ops? Is there a compile option for it?
- Impact: in training, magnitudes drift upward over the forward activations. LoRA gradients through 2 Llama-3.2-1B
  layers were 2–13 % off a CPU bf16 run.
