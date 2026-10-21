"""A100 reference for the energy comparison (#22): the same LoRA recipe as `npu-lora train`, in plain PyTorch.

The npu-lora CUDA backend runs the RNGD graph structure on the GPU (per-layer graphs, host AdamW), which is
not how one would train on a GPU. This script is the usual GPU way: the Hugging Face model in bf16 with
LoRA modules on q/v (PEFT initialisation), autograd, fused CUDA AdamW on fp32 adapter weights. Data,
packing (block-diagonal mask), schedule (warmup + cosine), clipping and seed follow npu-lora.

  python3 w7-native-lora-a100.py --model meta-llama/Llama-3.1-8B-Instruct --data nsmc_train.jsonl \
      --seq-len 128 --steps 200 --accum 4 --lr 2e-4 --log native.jsonl
Needs npu_lora on PYTHONPATH (data packing only) and transformers.
"""
import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from npu_lora import llama as L
from npu_lora.data import PackedDataset


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha, gen):
        super().__init__()
        self.base, self.scale = base, alpha / r
        d_out, d_in = base.weight.shape
        bound = 1.0 / math.sqrt(d_in)                    # kaiming_uniform(a=sqrt 5), as PEFT
        self.A = torch.nn.Parameter(((torch.rand(r, d_in, generator=gen) * 2 - 1) * bound).to(base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(d_out, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    path = L.resolve_model_path(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.requires_grad_(False)
    gen = torch.Generator().manual_seed(a.seed)
    params = []
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            m = LoRALinear(getattr(layer.self_attn, name), a.rank, a.alpha, gen)
            setattr(layer.self_attn, name, m)
            params += [m.A, m.B]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    ds = PackedDataset(a.data, tok, a.seq_len, seed=a.seed)
    log = open(a.log, "a") if a.log else None

    def lr_at(step):
        if step <= a.warmup:
            return a.lr * step / max(a.warmup, 1)
        prog = (step - a.warmup) / max(a.steps - a.warmup, 1)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    step = 0
    for batch in ds.batches(a.accum, a.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        losses, toks = [], 0
        for ids, labels, doc in batch:
            mask = L.causal_mask(len(ids), doc).to(torch.bfloat16).view(1, 1, len(ids), len(ids)).cuda()
            logits = model(input_ids=ids.view(1, -1).cuda(), attention_mask=mask).logits[0].float()
            lab = labels.cuda()
            n = int((lab >= 0).sum())
            loss = torch.nn.functional.cross_entropy(logits, lab.clamp(min=0), reduction="none")
            loss = (loss * (lab >= 0)).sum() / max(n, 1)
            (loss / len(batch)).backward()
            losses.append(float(loss))
            toks += len(ids)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        step += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rec = {"ts": round(time.time(), 3), "run": "native-lora-a100", "event": "step", "step": step,
               "loss": sum(losses) / len(losses), "grad_norm": gnorm, "lr": lr_at(step), "tokens": toks,
               "seconds": round(dt, 3), "tokens_per_s": round(toks / dt, 1)}
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
