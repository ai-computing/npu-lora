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

## Optimizer placement: step-time model

With resident host buffers the offload path is bound by the host, not by the PCIe links: splitting the
layers over two cards (two links) made the step slower on a 24-core host (6.36 → 9.26 s), where a model
that charges transfer bytes / link bandwidth predicted 5.6 s
([two-card record](../experiments/2026-10-21-two-card-split.md)). The model below charges host time instead.

Layers are either *offloaded* (fp32 master and AdamW state on the host) or *resident* (`--resident-layers N`:
the first N layers of each card keep fp32 master, m and v on the card and update with the `ResidentAdamW` graph).

```
T_step ≈ T_dev + T_res + T_wait
T_res  = N_res · t_dev                      device AdamW, main thread (8B layer: 28–33 ms)
T_wait = max(0, N_off · (t_a + t_u) / W  −  T_res  −  N_res,first · t_fwd)
```

- `T_dev`: forward + backward of the step on the device(s), including the hidden-state hops between cards.
  Measure it once as step − `wait_weights_s` of an offload run on the same cards.
- `t_a`, `t_u`: host AdamW and host→device upload time per offloaded layer, measured with W worker threads
  (`--update-workers`, default 4) as `prev_adamw_s / 32` and `prev_upload_s / 32` of the offload run. They are
  host throughput terms: on a contended or small host they grow, and adding cards does not shrink them.
- Gradient downloads overlap the backward pass and are not on the critical path. The layer jobs start
  when the global clip is known, and the pool runs them in backward order (layer 31 first), so the next
  forward waits for nearly all offloaded layers. Resident layers sit at the front of each card's range,
  so the next forward runs through them (`t_fwd` each) while the host jobs finish.

Check on RNGD, Llama-3.1-8B, S=512, accum 4 (host terms from an offload run on the same host at the same load;
records: [resident AdamW](../experiments/2026-10-21-resident-adamw.md), [two cards](../experiments/2026-10-21-two-card-split.md),
[DRAM_SHARED race](../experiments/2026-10-21-dram-shared-race.md)):

| host, configuration | predicted | measured | error |
| --- | ---: | ---: | ---: |
| 96 cores, 1 card, offload, W=8 (per-layer terms of the W=4 run) | 4.53 s | 4.82 s | −6 % |
| 96 cores, 1 card, 4 resident layers (steps 6–12, uncontended; that run stopped at 12, #40) | 5.10 s | 5.12 s | −0.4 % |
| 96 cores at load ~100, 1 card, 4 resident layers (steps 6–25; terms of the 1-card offload run right before) | 10.82 s | 9.72 s | +11 % |
| 96 cores, 2 cards, 8 resident layers per card (steps 6–25; terms of the 2-card offload run back to back) | 4.22 s | 4.27 s | −1 % |
| 24 cores, 2 cards, offload | 8.33 s | 9.26 s | −10 % |

The link-bandwidth model predicted the last row as 5.6 s (−40 %).

What the model says: once half of the layers are resident, `T_wait` is a few hundred ms and `T_dev`
(3.5 s on two cards) is the step: 2 cards × 8 resident layers run at 4.27 s/step, 15 % faster than 2-card offload (5.04 s). More resident layers only help until `T_wait` reaches zero; after that the
device compute path, not the optimizer, is the next limit. A card holds about 9 resident 8B layers next to
its share of bf16 weights and gradients, so all 32 layers resident needs 4 cards.
Resident runs used to stop with an intermittent `OutOfMemoryError ... on DRAM_SHARED` after 3–13 steps
([#40](https://github.com/ai-computing/npu-train/issues/40)). The cause was not memory: furiosa-torch 2026.2 keeps the
device memory type in one process-wide flag that every graph call briefly sets to DRAM_SHARED (0 bytes on a single-chip
card), so a layer worker's upload at that moment failed. Graph calls and device allocations now share one lock
(`backends.DEVICE_ALLOC_LOCK`; see [compiler workarounds](compiler-workarounds.md)).

## Job layout on Kubernetes

furiosa-torch opens a card only from a privileged container, which then sees every RNGD on the node.
A non-privileged init container records the card the device plugin assigned; the trainer maps that
name to furiosa-torch's ordinal (`rngd:N` is the N-th visible card). The official `furiosa-llm` image
lacks the aarch64 cross toolchain the compiler calls, so the Job installs it first.
