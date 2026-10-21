# Design

## Why training works on an inference-only NPU

An ahead-of-time NPU compiler assumes weights are constants baked into a forward-only graph.
npu-lora sidesteps both assumptions:

1. **Weights are graph inputs.** Frozen weights, LoRA adapters (or, in full fine-tuning, the weights
   being trained) are all passed in at call time. One decoder-layer graph is compiled once and reused
   for every layer by swapping the weight tensors.
2. **The backward pass is just another graph.** `torch.func.vjp` is traced with `make_fx` into a flat
   aten graph in which gradient ops are ordinary forward ops. The NPU runs it like an inference graph.
3. **The optimizer runs on the host** in fp32 (`torch.optim.AdamW`, fused CPU kernel).

The device runs these graphs per step:

| Graph | Inputs → outputs |
| --- | --- |
| layer forward | `h, weights, adapters, rope tables, mask → h'` |
| layer backward (recomputes forward = layer-level activation checkpointing) | `… , g_out → g_h, g_adapters` (LoRA) or `g_h, g_weights` (full) |
| layer backward + accumulate (full, accum > 1) | `… , g_out, acc → g_h, acc + g_weights` |
| head | RMSNorm, vocab-chunk statistics, vocab-chunk gradient, RMSNorm backward |

Each graph is split into parts of two matmuls (`graph.compile_parts`) because the compiler cannot
handle large fused graphs; parts exchange device tensors. Compiled parts are cached as EDF blobs keyed
by graph code and input strides.

## Vocabulary-chunked cross-entropy

`[S, 128k]` fp32 logits do not fit on-chip, so the head works on 32 vocabulary chunks. For each chunk
the device returns the row max, Σexp and the target logit; the host combines them into the
log-sum-exp; the device then accumulates `(softmax − onehot) · W_chunk` into the hidden gradient.

## Full fine-tuning: CPU-offload pipeline

Device memory holds only bf16 weights and the accumulated gradients. fp32 master weights and AdamW
state live in host memory (Llama-3.1-8B: 16 GB on device, ~128 GB on host).

```
step t   backward:  L31 ──► L30 ──► … ──► L0
                     │       │            │       (per layer, worker threads)
                     ▼       ▼            ▼
                    D2H     D2H   …      D2H  ─►  wait for global clip ─► AdamW ─► H2D
                                                                                   │
step t+1 forward:  L0 waits for its H2D ─► L1 waits ─► …                           ┘
```

- Gradients are accumulated on the device across micro-batches and downloaded once per step into
  host buffers that are allocated once per layer and reused (`Backend.host_buffer`, `to_host(out=)`).
  On RNGD, `t.cpu()` reaches only 3.0 GB/s because it allocates fresh host memory each time; copying
  into a reused buffer reaches 19.9 GB/s. On CUDA the buffers are pinned and each worker thread copies
  on its own stream, so copies do not queue behind compute.
- Each layer's squared gradient norm is reduced on the device right after its backward, so the
  global clip coefficient is known as soon as the backward pass ends — without waiting for the large
  downloads.
- Each layer's job (download → AdamW → upload) starts as soon as its gradients exist and runs in a
  worker pool; the next step's forward waits for a layer only when it reaches it. The resulting update
  is identical to a synchronous step (`tests/test_cpu.py::test_full_pipeline_equals_sync_update`).

Llama-3.1-8B full fine-tuning, S=512, one card (tokens/s):

| | accum 1 | accum 4 |
| --- | ---: | ---: |
| RNGD | 128 | 315 |
| A100 80GB | 153 | 477 |

The remaining ~3 s per step is shared between host AdamW over 7B parameters and the transfers.

## Job layout on Kubernetes

furiosa-torch opens a card only from a privileged container, which then sees every RNGD on the node.
A non-privileged init container records the card the device plugin assigned; the trainer maps that
name to furiosa-torch's ordinal (`rngd:N` is the N-th visible card). The official `furiosa-llm` image
lacks the aarch64 cross toolchain the compiler calls, so the Job installs it first.
