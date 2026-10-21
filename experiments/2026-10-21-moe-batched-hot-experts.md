# MoE hot experts on RNGD: one batched device call per layer vs per-expert calls vs host-only (Qwen3-30B-A3B)

- **Date**: 2026-10-21
- **Task**: #35
- **Host / device**: npu-node-1, one RNGD card per Job (k8s, npu0), `furiosaai/furiosa-llm:latest` (furiosa-torch 2026.2.0,
  furiosa-llm 2026.3.0, torch 2.10.0+cpu, transformers 5.1.0). Pods requested 32 CPUs. The node is shared: its load average was
  95–380 during these runs, 120–150 during the A/B runs.
- **Commit**: batched `perf/35-batch-hot-experts` (`961f4ec` + upload counters `4057701`). Per-expert baseline: `main` `a07aefc`.

## Question

#27 measured about 615 s/step with hot experts on RNGD against about 195 s/step host-only (another host, OptimusPrime driver),
with one device call per hot expert. Does one batched call per layer make the device path at least as fast as host-only?

## Setup

- Qwen/Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 experts, top-8, H 2048, expert F 768). LoRA r=16 α=32 on q·v, S=128,
  accum 1, lr 2e-4, NSMC train data, seed 0.
- `--moe-hot-fraction 0.25` gives 32 hot experts per layer, the most-routed ones after 4 host-only warm-up steps.
- Batched graphs per bucket (16/32/64/128 rows per expert, E=32): about 40 min of CPU compile per bucket, compiled
  ahead in CPU Jobs (`moe_precompile.py`). The per-expert graphs for `main` were compiled the same way (2–5 min per bucket).
- **A/B method** (`moe_bench2.py`): one process and one model load. After the warm-up, steps alternate hot / host-only (12 + 12),
  so the host load drift hits both sides. The hot set is fixed after the warm-up.

## Command

```bash
python3 -m npu_lora.moe_bench2 Qwen/Qwen3-30B-A3B-Instruct-2507 /mnt/store/datasets/quality/nsmc_train.jsonl 0.25 out.jsonl 24
python3 -m npu_lora.moe_micro micro.json        # one batched call vs the host per-expert path, random weights
```

## Results

### Step time, alternating steps (steps 6–28, same process)

| code | hot-expert step median (mean) | host-only step median (mean) | device calls / step | hot / host |
|---|---:|---:|---:|---:|
| `main`, per-expert calls | 42.4 s (41.0) | 17.2 s (17.6) | 3,068 | **2.46×** |
| **batched (this branch)** | **13.0 s** (12.2) | 13.5 s (13.0) | 144 | **0.97×** |

- Device calls per step: 48 layers × (forward, the forward that `HostMLP.backward` recomputes, backward).
- The paired step differences (hot minus the next host-only step) for the batched code have a median of −0.4 s. The batched path is at parity with host-only, not clearly faster. On `main` the gap is +23.7 s.

### Micro-benchmark: one batched call (E=32, H 2048, F 768)

| rows per expert | upload | forward | backward | download | host per-expert fwd+bwd for the same rows |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.9 ms | 2.3 ms | 4.7 ms | 1.5 ms | 56 ms |
| 32 | 0.9 ms | 3.6 ms | 5.6 ms | 3.5 ms | 46 ms |
| 64 | 2.6 ms | 6.3 ms | 9.0 ms | 7.8 ms | 58 ms |
| 128 | 3.4 ms | 12.0 ms | 10.2 ms | 12.7 ms | 65 ms |

- One batched call is several times faster than the host for the same rows.
- At S=128 the hot experts carry only part of the routed rows. The rest of the step is the cold experts, router and norms on a shared host, and that host work sets the step time.

### Re-planning the hot set every step costs uploads

`moe_bench2` phases (`mb-phases-batched.jsonl`):
- With `moe_replan_every=1`, 11–47 of 48 layers changed their hot set per step.
- Each change re-uploads that layer's stacked weights (302 MB): 1.2–7.0 s per step.
- With a fixed hot set: no uploads after the first step.

Raw data in `experiments/data/moe-batched/`:
- `mb-abab-batched.jsonl`;
- `mb-abab-per-expert-main.jsonl`;
- `mb-phases-batched.jsonl`;
- `moe-micro.json`;
- earlier sequential runs: `mb-host-first.jsonl`, `mb-hot-then-host.jsonl` (host load differed between them);
- the scripts.

## Conclusion

- Batching the hot experts of a layer into one device call removes the per-expert overhead: 3,068 → 144 device calls per step, and hot-expert steps go from 2.46× host-only to 0.97× on the same host and load.
- It does not yet make hot experts clearly faster than host-only at S=128: device work is a few ms per layer, and the step is dominated by host work.
- Use a fixed or rarely re-planned hot set (`moe_replan_every` large). Re-planning every step re-uploads stacked weights.
- Still open for #35's "faster than host-only":
  - longer sequences or more micro-batch rows per call, so the hot experts carry more rows;
  - a larger hot fraction (E=64 graphs, about 29 GB of stacked weights on the card);
  - removing the forward that `HostMLP.backward` recomputes (one of the three calls per layer).
- The losses of both paths match the CPU equivalence test (`test_moe_hot_experts_on_device_match_host`). Per-step losses on RNGD are not compared here: the head on `main` rounds the target logit (#33, fixed in #43), and several steps report small negative losses on both paths.

## Follow-up: profile, then drop the recomputed forward (`HostMLP.keep_graph`)

**Commit**: `perf/35-hot-experts-faster` `1fee9f2`. Same host, card and recipe as above, S=128, hot fraction 0.25.

**Method** (`moe_prof.py`, raw data `prof-s128-roundrobin.jsonl`):
- One process. 4 host-only warm-up steps fix the hot set.
- Then 6 rounds of the four configurations, with the order reversed every other round.
- Load average 53 during this run, lower than during the A/B runs above (120–150).

| config (step median, 5–6 steps each) | step | attention + head on device | HostMLP fwd | HostMLP bwd | of which: recomputed fwd | cold-expert fwd | hot device fwd / bwd |
|---|---:|---:|---:|---:|---:|---:|---:|
| host-only (as `main`) | 9.21 s | 1.26 | 2.48 | 5.34 | 2.48 | 2.46 | — |
| host-only + keep_graph | 8.00 s | 1.16 | 2.46 | 4.31 | 0 | 1.31 | — |
| hot (as #38) | 6.70 s | 1.07 | 1.73 | 3.79 | 1.66 | 1.27 | 0.55 / 0.52 |
| **hot + keep_graph** | **5.19 s** | 1.10 | 1.81 | 2.21 | 0 | 0.64 | 0.42 / 0.59 |

- The cold-expert forward column counts both the forward and the recompute, so it halves when the recompute is dropped.
- The router takes ≤ 0.05 s per step.
- `HostMLP.backward` recomputed the whole MoE block forward: about a third of the host MoE time, and one of the three device calls per layer.
- `keep_graph` keeps the forward's autograd graph until that layer's backward. It is bitwise-equal to recomputing (`test_host_mlp_keep_graph_equals_recompute`) and costs a few MB of activations per layer at S=128.

**Paired per round** (hot + keep_graph minus the host-only step of the same round):

| baseline | per-round differences | median | rounds where hot is faster |
|---|---|---:|---:|
| vs host-only + keep_graph | −1.33, −3.13, −3.84, −2.50, −2.81, −1.25 s | **−2.66 s (−33%)** | 6 / 6 |
| vs host-only as on `main` | −4.71, −5.03, −4.60, −2.58, −3.74, −3.63 s | **−4.17 s (−45%)** | 6 / 6 |

**Conclusion**:
- With the recompute dropped, hot experts on RNGD are faster than host-only by more than the round-to-round noise at this load: 6 of 6 rounds, median −2.66 s against the equally improved host-only path.
- Even without `keep_graph`, hot (6.70 s) beat host-only (9.21 s) at load 53. In the A/B above, at load 120–150, they were at parity. The gain depends on how busy the shared host is, because the cold experts and the backward run there.
- Not tried: E=64 and S=256/512. Both need about 40 min of compile per expert bucket, and S needs new attention graphs too. They were not needed to answer #35.
