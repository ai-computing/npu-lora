# Qwen2 on RNGD: why `npu-lora check` failed, and the fix (bf16 matmul inputs, large q/k bias, head target logit)

- **Date**: 2026-10-21
- **Task**: #33
- **Host / device**: npu-node-1, one RNGD card per Job (npu0 or npu1, assigned by k8s), `furiosaai/furiosa-llm:latest`
  (furiosa-torch 2026.2.0, furiosa-llm 2026.3.0, torch 2.10.0+cpu, transformers 5.1.0). Compile jobs ran on CPU nodes.
  CPU references ran on a Windows PC (torch 2.14.1+cpu, transformers 5.20.2).
- **Commit**: baseline `main` (`cf3fb32`). Fix: `fix/33-qwen-lora-grad` (final graph `1bcae90`).

## Question

#31 found that Qwen2.5-1.5B fails `npu-lora check` on RNGD: worst LoRA gradient error 0.444 (layer 0 q B, cos 0.902), v 0.13–0.16.
11 of 400 training steps reported a negative cross-entropy. Which graph part is wrong, and can npu-train fix it?

## Setup

The check setup: 2 layers, S=128, random token ids, LoRA r=16 q·v, B ~ N(0, 0.01).
Three references are used:
- RNGD;
- the same graphs in bf16 on the CPU (what `check` compares against);
- an **fp32** autograd reference with the same bf16 weights (`data/qwen2-precision/sens.py`, `diag33.py`).

`diagnose.compare_parts` ran each compiled part on RNGD and on the CPU with the real layer-0 inputs.

## Results

### 1. The CPU reference itself was wrong: Qwen2's q/k bias makes the bf16 gradient inaccurate

| `main`, relative error of the LoRA grads (layer 0 q A, q B, v A, v B) | vs fp32 |
|---|---|
| CPU, bf16 graph | 0.537, 0.541, 0.201, 0.210 |
| RNGD | 0.517, 0.507, 0.193, 0.194 |
| RNGD vs CPU-bf16 (what `check` reported) | 0.417, 0.443, 0.137, 0.155 |

- `compare_parts` on `main`: every part is within 1% of the CPU on identical inputs. The compiler does not produce wrong values here; both bf16 runs are far from fp32.
- Cause: Qwen2.5's q/k biases are large next to `x @ W` (|q|, |k| reach the hundreds). bf16 `q + bq` and `q·k` keep 2–3 significant digits of the attention logits.
- On the CPU (eager bf16 with chosen ops in fp32, `sens2`):

  | variant | worst rel. error vs fp32 |
  |---|---:|
  | bf16 | 0.549 |
  | only q·kᵀ in fp32 | 0.273 |
  | q/k path (projection output, bias, RoPE, q·kᵀ) in fp32 | **0.016** |

### 2. RNGD reads bf16 matmul inputs even in fp32 graphs

With q/k in fp32 the CPU-bf16 graph came to 1.6% of fp32, but RNGD stayed at 0.40 / 0.37 (q) and 0.05 (v).
- `compare_parts` now flagged one part: the backward `dq = dS @ k` (fp32 bmm), 13.9% off.
- `dq_i = Σ_j dS_ij k_j` with `Σ_j dS_ij = 0` cancels around the large bias part of k.
- A CPU emulation that rounds the attention matmul inputs to bf16 (fwd and bwd, fp32 accumulate; `sens4.py`) reproduces RNGD:

  | emulated RNGD matmul | q A | q B | v A | v B | (layer 1) |
  |---|---:|---:|---:|---:|---|
  | fp32 q/k path | 0.371 | 0.345 | 0.055 | 0.069 | 0.11 / 0.10 / 0.056 / 0.057 |
  | + `k − mean_j(k_j)` | **0.012** | **0.011** | 0.004 | 0.004 | 0.009 / 0.008 / 0.004 / 0.004 |
  | RNGD measured (fp32 q/k path) | 0.405 | 0.373 | 0.053 | 0.067 | 0.111 / 0.100 / 0.053 / 0.053 |

- `softmax(q_i·(k_j − k̄)) = softmax(q_i·k_j)`, because `q_i·k̄` is the same for every key of row i. Centering k is exact and removes the common part before any matmul sees it.
- `k̄` is detached: its exact gradient is zero (`Σ_j dS_ij = 0`), and the furiosa compiler rejects the reduction its backward would add (`ReduceByVe` verification).
- Taking the mean over the transposed `[nkv, S, hd]` view compiled but the run failed with `failed to run TP program … TimedOut`. This reproduced on two cards (fwd part 11 of the layer, commit `a83b1a8`; `data/qwen2-precision/runparts.py`). A standalone graph with the same ops did **not** reproduce it.
- Taking the mean over `[S, nkv, hd]` before the transpose runs.

### 3. Negative loss: RNGD rounds the target-logit reduction

The head probe (`headprobe.py`) used one 4096-token chunk with confident rows and float64 truth. Errors are maxima over rows.

| chunk graph | m error | t error | lse error | rows with lse − t < 0 |
|---|---:|---:|---:|---:|
| bf16 logits (`main`) | 0.119 | 0.119 | 0.119 | 0 |
| fp32 logits | 1.3e-5 | **0.119** | 1.3e-5 | **46 / 128** |
| fp32 logits, m rounded to bf16 | 1.3e-5 | 0.119 | 1.3e-5 | 46 |

- `(logits · onehot).sum` is computed from bf16-rounded logits even when the graph says fp32, while `amax` and `Σ exp` use fp32.
- In the real head (`main`) the target logit exceeded its row max by up to 0.06, which gives `lse − t < 0` on confident tokens.
- Fix: the chunk graphs return only `(m, Σ exp)` from fp32 logits. The target logit is an S-row dot product on the host (`llama.target_logits`).

### 4. Final graph (`1bcae90`) on RNGD

`npu-lora check --layers 2 --seq-len 128`: **pass**.
- Loss rel. error 3.0e-4.
- Worst gradient 3.13% (layer 0 q B, cos 0.99951). The other tensors are 1.2–3.0%, cos ≥ 0.9995.

Against fp32, the RNGD gradients are 0.9–2.4% off and the CPU-bf16 ones 0.9–1.7% off.

`compare_parts`: no part above 1% (16 fwd, 40 bwd).

The head on confident rows had no negative row, and `t − max_row ≤ 4e-6`.

LoRA training, the same recipe as #31 (Qwen2.5-1.5B-Instruct, NSMC 4,000, r16 q·v, S128, 400 steps × accum 4, lr 2e-4, seed 0):

| run | negative steps | min loss | loss step 1–3 | step 100 | step 400 | mean of last 40 | median step |
|---|---:|---:|---|---:|---:|---:|---:|
| RNGD `main` (#31) | 11 | −0.012 | 0.495 / 0.346 / 2.079 | 0.124 | 0.004 | 0.106 | 4.58 s |
| **RNGD fixed** | **0** | 0.0015 | 0.473 / 0.366 / 2.023 | 0.071 | 0.018 | 0.091 | 2.74 s |
| A100 (#31) | 0 | 0.002 | 0.489 / 0.362 / 2.060 | 0.073 | 0.017 | 0.096 | 0.43 s |

The fixed RNGD curve follows the A100 one (step 100: 0.071 vs 0.073). The `main` RNGD curve did not (0.124).

Step times come from a shared host (load average 130–380 during these runs). The faster RNGD step is not a controlled comparison.

Raw data in `experiments/data/qwen2-precision/`:
- `check-final-rngd.json`;
- `diag-{main,fp32qk,final}-rngd.json` (the first two use the earlier head diagnostic, which read t from the chunk graph);
- `headprobe-rngd.json`;
- `qwen15-lora-rngd-fixed.jsonl`;
- the scripts.

## Command

```bash
M=Qwen/Qwen2.5-1.5B-Instruct
python3 -m npu_lora.cli compile --model $M --seq-len 128 --cache-dir /cache/furiosa            # CPU node
python3 -m npu_lora.cli check --model $M --backend furiosa:auto --cache-dir /cache/furiosa --seq-len 128 --layers 2
python3 -m npu_lora.diag33 $M diag.json 2 128          # data/qwen2-precision/diag33.py copied into npu_lora/
python3 -m npu_lora.cli train --model $M --data /mnt/store/datasets/quality/nsmc_train.jsonl --backend furiosa:auto \
  --cache-dir /cache/furiosa --seq-len 128 --steps 400 --accum 4 --lr 2e-4 --rank 16 --alpha 32 --targets q,v --seed 0 \
  --run-name nta-qwen15-rngd-fix --out /mnt/store/nta/out/qwen15-rngd-fix --log-file qwen15-rngd-fix.jsonl
python sens.py <repo> $M 2 128; python sens4.py <repo> $M 2 128                                 # CPU
```

## Conclusion

The #33 failure was numerical, not a miscompiled part:
- Qwen2's large q/k biases need more than bf16 in the attention logits.
- RNGD matmuls read bf16 inputs regardless of the graph dtype.

In npu-train, three changes fix it:
- fp32 q/k from the bias add;
- k centred over positions (exact, detached);
- the head target logit computed on the host.

With them `npu-lora check` passes (3.1%), no step reports a negative loss, and the loss curve matches A100.

The head change applies to every model. Llama-3.1-8B `npu-lora check --layers 2 --seq-len 128` on RNGD passes both before and after:
- `main`: loss rel 4.2e-5, worst gradient 2.22% (cos 0.99976);
- this branch: loss rel 1.8e-5, worst gradient 2.14% (cos 0.99977).

Raw data: `llama8b-check-{main,final}-rngd.json`.

Compiler behaviour worth reporting to the vendor:
- fp32 matmul inputs are rounded to bf16;
- `(x·onehot).sum` over fp32 logits uses bf16 values;
- a fwd part with a positions-mean on a transposed view compiles but times out at run time.

Next: Qwen3 (no q/k bias; its check already passed for Qwen3-0.6B in #32).
