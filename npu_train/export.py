"""학습 결과 내보내기.

- save_peft: HF PEFT 형식(adapter_config.json + adapter_model.safetensors). vLLM(--enable-lora),
  HF peft, llm-d 의 LoRA 라우팅이 그대로 읽는다.
- save_merged: W' = W + scale·B·A 를 원래 가중치에 합친 전체 모델. LoRA 를 런타임에 못 붙이는
  NPU 엔진(artifact 를 미리 컴파일하는 Furiosa/RBLN)은 이걸 다시 컴파일해 서빙한다.
"""
import json
import os
import shutil

import torch

from . import llama as L

PEFT_MODULE = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj"}
WEIGHT_KEY = {"q": "wq", "k": "wk", "v": "wv", "o": "wo"}


def save_peft(trainer, out_dir, base_model_name=None, extra=None):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for (i, t), (a, b) in trainer.adapters().items():
        p = f"base_model.model.model.layers.{i}.{PEFT_MODULE[t]}"
        tensors[f"{p}.lora_A.weight"] = a.to(torch.bfloat16).contiguous()
        tensors[f"{p}.lora_B.weight"] = b.to(torch.bfloat16).contiguous()
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name or trainer.path,
        "r": trainer.lora.r, "lora_alpha": trainer.lora.alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": [PEFT_MODULE[t].split(".")[-1] for t in trainer.lora.targets],
        "fan_in_fan_out": False, "inference_mode": True,
    }
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)
    meta = {"trained_with": "npu-lora", **trainer.summary(), **(extra or {})}
    json.dump(meta, open(os.path.join(out_dir, "npu_lora_meta.json"), "w"), indent=2, default=str)
    return out_dir


def load_peft(path):
    """{(layer, target): (A, B)}, r, alpha"""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    t = load_file(os.path.join(path, "adapter_model.safetensors"))
    inv = {v.split(".")[-1]: k for k, v in PEFT_MODULE.items()}
    out = {}
    for k, a in t.items():
        if not k.endswith("lora_A.weight"):
            continue
        parts = k.split(".")
        i = int(parts[parts.index("layers") + 1])
        tgt = inv[parts[-3]]
        out[(i, tgt)] = (a.float(), t[k.replace("lora_A", "lora_B")].float())
    return out, cfg["r"], cfg["lora_alpha"]


def save_merged(base_path, adapter_path, out_dir):
    """base 에 어댑터를 합친 safetensors 를 out_dir 에 쓴다(설정·토크나이저 파일은 복사)."""
    from safetensors.torch import save_file
    base_path = L.resolve_model_path(base_path)
    ad, r, alpha = load_peft(adapter_path)
    scale = alpha / r
    W = L.Weights(base_path)
    os.makedirs(out_dir, exist_ok=True)
    by_file = {}
    for name, fn in W.where.items():
        by_file.setdefault(fn, []).append(name)
    for fn, names in by_file.items():
        part = {}
        for n in names:
            w = W.get(n, dtype=torch.float32)
            for (i, t), (a, b) in ad.items():
                if n == f"model.layers.{i}.{PEFT_MODULE[t]}.weight":
                    w = w + scale * (b @ a)
            part[n] = w.to(torch.bfloat16).contiguous()
        save_file(part, os.path.join(out_dir, fn), metadata={"format": "pt"})
    for f in os.listdir(base_path):
        if not f.endswith(".safetensors"):
            src = os.path.join(base_path, f)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out_dir, f))
    return out_dir


def save_full(trainer, out_dir, extra=None):
    """전체 파인튜닝 결과를 HF 형식 전체 모델로 쓴다(base 의 나머지 텐서·설정·토크나이저는 그대로).

    vLLM·HF 가 바로 읽는다. NPU 서빙(Furiosa/RBLN)은 이 디렉터리를 다시 컴파일한다.
    """
    from safetensors.torch import save_file
    base = trainer.path
    W = L.Weights(base)
    new = {f"model.layers.{i}.{L.hf_name(k)}": w for (i, k), w in trainer.layer_weights().items()}
    os.makedirs(out_dir, exist_ok=True)
    if trainer.host_mlp is not None:     # MoE: the checkpoint's expert layout differs from the modules' (fused)
        return _save_full_hf(trainer, base, out_dir, new, extra)
    by_file = {}
    for name, fn in W.where.items():
        by_file.setdefault(fn, []).append(name)
    for fn, names in by_file.items():
        part = {n: (new[n] if n in new else W.get(n, dtype=torch.float32)).to(torch.bfloat16).contiguous()
                for n in names}
        save_file(part, os.path.join(out_dir, fn), metadata={"format": "pt"})
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if not f.endswith(".safetensors") and os.path.isfile(src):
            shutil.copy(src, os.path.join(out_dir, f))
    meta = {"trained_with": "npu-lora", "mode": "full", **trainer.summary(), **(extra or {})}
    json.dump(meta, open(os.path.join(out_dir, "npu_lora_meta.json"), "w"), indent=2, default=str)
    return out_dir


def _save_full_hf(trainer, base, out_dir, new, extra):
    """MoE: load the base model, put in the trained attention weights and the trained host blocks, and let
    transformers write the checkpoint (it owns the mapping between fused expert tensors and the file layout)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    state = dict(model.named_parameters())
    with torch.no_grad():
        for name, w in new.items():
            state[name].copy_(w.to(torch.bfloat16))
        for i in range(len(trainer.host_mlp.blocks)):
            for name, p in trainer.host_mlp.named_parameters(i):
                state[name].copy_(p.detach().to(torch.bfloat16))
    model.save_pretrained(out_dir)
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if not f.endswith((".safetensors", ".json")) and os.path.isfile(src) or f in (
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
            if os.path.isfile(src) and not os.path.exists(os.path.join(out_dir, f)):
                shutil.copy(src, os.path.join(out_dir, f))
    meta = {"trained_with": "npu-lora", "mode": "full", **trainer.summary(), **(extra or {})}
    json.dump(meta, open(os.path.join(out_dir, "npu_lora_meta.json"), "w"), indent=2, default=str)
    return out_dir
