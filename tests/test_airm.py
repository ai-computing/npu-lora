"""ai-rm adapter 테스트 — NPU 없이 돈다. 패키지를 설치(`pip install -e .`)해야 entry point 가 보인다.

  python tests/test_airm.py    # 또는 python -m pytest tests/
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rm_core import ContractError, Workload, prometheus_name, registry  # noqa: E402
from rm_core.conformance import check_measurements  # noqa: E402

WORDS = [f"w{i}" for i in range(200)]


def _tiny_model():
    """2층 Llama + 단어 단위 토크나이저를 임시 디렉터리에 쓴다(네트워크 없이)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    d = tempfile.mkdtemp()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=2, vocab_size=256, rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d, safe_serialization=True)
    vocab = {"[UNK]": 0, "[EOS]": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(d)
    data = os.path.join(d, "train.jsonl")
    with open(data, "w") as f:
        for i in range(8):
            f.write(json.dumps({"text": " ".join(WORDS[(i * 7 + j) % 200] for j in range(20))}) + "\n")
    return d, data


def test_registry_loads_runner():
    assert registry.available("airm.workloads").get("npu-lora") == "npu_lora.airm:train"
    run = registry.load("airm.workloads", "npu-lora")
    assert callable(run) and run.name == "npu-lora"


def test_cpu_train_tiny():
    model, data = _tiny_model()
    out = os.path.join(model, "adapter")
    run = registry.load("airm.workloads", "npu-lora", backend="cpu")
    res = run(Workload(kind="train", model=model), data=data, out=out, dataset="tiny", steps=2, seq_len=32,
              rank=4, alpha=8, warmup=1, run_name="airm-test")
    assert res["steps"] == 2 and res["mode"] == "lora" and res["run"] == "airm-test"
    assert os.path.exists(os.path.join(out, "adapter_model.safetensors"))
    check_measurements(res["measurements"])
    (m,) = res["measurements"]
    assert m.name == "tok_s" and m.value > 0
    assert m.tags == {"component": "npu-lora", "workload": "train", "model": os.path.basename(model),
                      "vendor": "cpu", "run_id": "airm-test", "dataset": "tiny"}


def test_rejects_wrong_workload_and_config():
    run = registry.load("airm.workloads", "npu-lora", backend="cpu")
    for wl, cfg in [(Workload(kind="serve", model="m"), {"data": "d", "out": "o"}),
                    (Workload(kind="train", model="m", precision="fp8"), {"data": "d", "out": "o"}),
                    (Workload(kind="train", model="m"), {"data": "d", "out": "o", "kv_transport": "p2p"}),
                    (Workload(kind="train", model="m"), {"data": "d"}),
                    (Workload(kind="train", model="m"), {"data": "d", "out": "o", "backend": "tpu"}),
                    (Workload(kind="train", model="m"), {"data": "d", "out": "o", "ranks": 4})]:
        try:
            run(wl, **cfg)
        except ContractError:
            continue
        raise AssertionError(f"accepted {wl} {cfg}")


def test_prometheus_uses_airm_name_and_tag_keys():
    from rm_core.metrics import TAG_KEYS
    from npu_lora.metrics import Reporter
    rep = Reporter("r", airm_tags={"component": "npu-lora", "workload": "train", "run_id": "r"})
    rep.step({"step": 1, "loss": 1.0, "grad_norm": 1.0, "lr": 1e-4, "tokens": 32, "seconds": 0.5,
              "tokens_per_s": 64.0})
    lines = [l for l in rep._render().splitlines() if l.startswith(prometheus_name("tok_s") + "{")]
    assert lines == ['airm_tok_s{component="npu-lora",workload="train",run_id="r"} 64.0']
    assert set(rep.airm_tags) <= set(TAG_KEYS)
    # deprecated 이름은 프로토콜 0.2 전까지 같은 값으로 남는다(PROTOCOL §6 이행 규칙)
    assert 'npu_lora_tokens_per_second{run="r"} 64.0' in rep._render().splitlines()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(e)[:200])
    sys.exit(1 if failed else 0)
