"""CPU 회귀 테스트 — NPU 없이 돈다. `python -m pytest tests/` 또는 `python tests/test_cpu.py`.

이번 개발에서 조용히 틀렸던 것들을 고정한다: 그래프 변환·분할이 값을 바꾸지 않는지, 같은 텐서 객체를
두 자리에 넘겨도 입력이 묶이지 않는지, 어휘 조각 cross-entropy 가 통째 계산과 같은지, 함수형 Llama 가
HF 와 같은지(transformers 가 있을 때).
"""
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from npu_lora import graph as G, llama as L  # noqa: E402
from npu_lora.backends import CPUBackend  # noqa: E402

BF = torch.bfloat16
C = L.LlamaConfig(H=64, I=128, nh=4, nkv=2, n_layers=2, vocab=1000)


def _layer_inputs(S=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s, sc=0.05: (torch.randn(*s, generator=g) * sc).to(BF)
    w = [torch.ones(C.H, dtype=BF), torch.ones(C.H, dtype=BF), rn(C.H, C.H), rn(C.kv_dim, C.H), rn(C.kv_dim, C.H),
         rn(C.H, C.H), rn(C.I, C.H), rn(C.I, C.H), rn(C.H, C.I)]
    ad = [rn(4, C.H), rn(C.H, 4), rn(4, C.H), rn(C.kv_dim, 4)]
    cos, sin = L.rope_tables(C, S)
    return [rn(S, C.H, sc=1.0), *w, *ad, cos, sin, L.causal_mask(S), L.rotate_half_matrix(C.hd)], rn(S, C.H, sc=1.0)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def test_llama3_rope_scaling_matches_hf():
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from transformers import LlamaConfig
    except ImportError:
        return
    hc = LlamaConfig(hidden_size=2048, num_attention_heads=32, rope_theta=500000.0,
                     rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](hc, "cpu")
    c = L.LlamaConfig(H=2048, I=8192, nh=32, nkv=8, n_layers=1, vocab=10, rope_scaling=(32.0, 1.0, 4.0, 8192))
    assert torch.allclose(L.rope_inv_freq(c), ref.float(), rtol=1e-5)


def test_full_bwd_matches_autograd():
    inp, gout = _layer_inputs()
    nw = len(L.LAYER_WEIGHTS)
    base = inp[:1 + nw] + inp[1 + nw + 4:]          # 어댑터 4개를 뺀다
    out = G.build(G.layer_bwd_full_fn(C), base + [gout])(*base, gout)
    h = base[0].float().requires_grad_()
    ws = [w.float().requires_grad_() for w in base[1:1 + nw]]
    y = L.layer(C, h, dict(zip(L.LAYER_WEIGHTS, ws)), {}, 1.0, *[t.float() for t in base[1 + nw:]])
    y.backward(gout.float())
    for a, r in zip(out, [h.grad] + [w.grad for w in ws]):
        assert _rel(a, r) < 3e-2


def test_full_pipeline_equals_sync_update():
    """offload 파이프라인(장치 누적·비동기 내리기·레이어별 갱신)이 동기 계산과 같은 가중치를 낸다."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    from npu_lora import LoRATrainer, LoRAConfig, TrainConfig
    d = tempfile.mkdtemp()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=2, vocab_size=1000, rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(BF).save_pretrained(d, safe_serialization=True)
    tc = TrainConfig(seq_len=32, full=True, accum=2, lr=1e-3, warmup=1, steps=4, grad_clip=0.5)
    pipe = LoRATrainer(d, CPUBackend(), LoRAConfig(), tc, log=lambda s: None)
    pipe.setup()
    ref = LoRATrainer(d, CPUBackend(), LoRAConfig(), tc, log=lambda s: None)
    ref.setup()
    g = torch.Generator().manual_seed(3)
    batches = []
    for _ in range(2):
        mb = []
        for _ in range(2):
            ids = torch.randint(0, 1000, (32,), generator=g)
            lab = torch.roll(ids, -1)
            lab[-1] = -1
            mb.append((ids, lab, None))
        batches.append(mb)
    for b in batches:
        pipe.train_step(b)
        # 기준: 같은 trainer 구조로 동기 계산 — 기울기 평균·전역 clip·AdamW
        grads = None
        for ids, lab, doc in b:
            _, gr, _ = ref.forward_backward(ids, lab, doc)
            grads = gr if grads is None else [[x + y for x, y in zip(a, c)] for a, c in zip(grads, gr)]
        grads = [[x / len(b) for x in gl] for gl in grads]
        gn = math.sqrt(sum(float(x.pow(2).sum()) for gl in grads for x in gl))
        clip = min(1.0, tc.grad_clip / (gn + 1e-6))
        ref.step_no += 1
        lr = ref._lr()
        for i in range(ref.c.n_layers):
            ref.dev_W[i] = ref._update_layer(i, [x.clone() for x in grads[i]], clip, lr)
    pipe.sync_params()
    for a, b in zip(pipe.params, ref.params):
        assert _rel(a.detach(), b.detach()) < 1e-2, _rel(a.detach(), b.detach())


def test_rope_mm_equals_rope():
    x = torch.randn(4, 16, 32)
    cos, sin = torch.randn(16, 32), torch.randn(16, 32)
    assert torch.allclose(L.rope_mm(x, cos, sin, L.rotate_half_matrix(32, torch.float32)), L.rope(x, cos, sin), atol=1e-5)


def test_passes_keep_values_and_parts_match_whole():
    inp, gout = _layer_inputs()
    fn = G.layer_bwd_fn(C, ("q", "v"), 2.0)
    ref = fn(*inp, gout)
    gm = G.build(fn, inp + [gout])          # build 은 변환 전후를 비교하고 다르면 예외
    whole = gm(*inp, gout)
    split = G.compile_parts(CPUBackend(), "t", gm, inp + [gout], max_mm=1, log=lambda m: None)
    parts = split(*inp, gout)
    for r, a, b in zip(ref, whole, parts):
        assert _rel(a, r) < 1e-3 and _rel(b, r) < 1e-3


def test_duplicate_sample_tensor_is_not_merged():
    f = lambda x, a, b: x * a + b
    x, one = torch.randn(4, 4), torch.ones(4, 4)
    gm = G.build(f, [x, one, one])           # 같은 객체를 두 자리에 — 그래프 입력이 묶이면 안 된다
    a, b = torch.full((4, 4), 2.0), torch.full((4, 4), 5.0)
    assert torch.allclose(gm(x, a, b), x * 2 + 5)


def test_materialize_outputs_makes_contiguous():
    gm = G.build(lambda w: (w * 2, w.t()), [torch.randn(3, 5)])
    with torch.no_grad():
        n = G.materialize_outputs(gm, gm(torch.randn(3, 5)))
    assert n == 1
    out = gm(torch.randn(3, 5))
    assert all(o.is_contiguous() for o in out)


def test_split_disconnected_components():
    gm = G.build(lambda x, y: (x @ x, y + 1), [torch.randn(4, 4), torch.randn(4, 4)])
    assign = G.plan_parts(gm, max_mm=5)
    assert len(set(assign.values())) == 2     # 이어지지 않은 두 계산은 다른 조각


def test_vocab_chunked_ce_matches_full():
    g = torch.Generator().manual_seed(0)
    S, V, H = 16, 1000, 64
    vc = 250
    x = torch.randn(S, H, generator=g).to(BF)
    lm = (torch.randn(V, H, generator=g) * 0.05).to(BF)
    labels = torch.randint(0, V, (S,), generator=g)
    labels[:3] = -1
    lab = labels.float().view(-1, 1)
    n = int((labels >= 0).sum())
    chunks = [(lm[j * vc:(j + 1) * vc], torch.zeros(1, vc), torch.arange(j * vc, (j + 1) * vc, dtype=torch.float32).view(1, -1))
              for j in range(V // vc)]
    stats = [L.ce_chunk_stats(x, *c, lab) for c in chunks]
    lse, t = L.ce_combine(stats)
    w = (labels >= 0).float().view(-1, 1) / n
    loss = float(((lse - t) * w).sum())
    gx = torch.zeros(S, H)
    for c in chunks:
        gx = L.ce_chunk_grad(x, *c, lab, lse, w, gx)
    xr = x.float().requires_grad_()
    lp = torch.log_softmax(xr @ lm.float().t(), -1)
    ref = -(lp[labels >= 0].gather(1, labels[labels >= 0].view(-1, 1))).sum() / n
    ref.backward()
    assert abs(loss - float(ref)) / float(ref) < 1e-2
    assert _rel(gx, xr.grad) < 2e-2


def test_vocab_chunks_divides_or_pads():
    assert L.vocab_chunks(128256) == (32, 4008, 0)
    n, vc, pad = L.vocab_chunks(151937)
    assert n * vc - pad == 151937 and pad > 0


def test_hf_equivalence():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    from npu_lora import LoRAConfig, LoRATrainer, TrainConfig, export
    d = tempfile.mkdtemp()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=2, vocab_size=1000, rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(BF).save_pretrained(os.path.join(d, "base"), safe_serialization=True)
    tr = LoRATrainer(os.path.join(d, "base"), CPUBackend(), LoRAConfig(r=4, alpha=8, targets=("q", "k", "v", "o")),
                     TrainConfig(seq_len=32), log=lambda s: None)
    tr.setup()
    g = torch.Generator().manual_seed(1)
    for per in tr.master:
        for b in per[1::2]:
            b.detach().normal_(0, 0.05, generator=g)
    tr._upload_adapters()
    ids = torch.randint(0, 1000, (32,), generator=g)
    lab = torch.roll(ids, -1)
    lab[-1] = -1
    loss, grads, _ = tr.forward_backward(ids, lab)
    export.save_peft(tr, os.path.join(d, "ad"))
    export.save_merged(os.path.join(d, "base"), os.path.join(d, "ad"), os.path.join(d, "merged"))
    m = LlamaForCausalLM.from_pretrained(os.path.join(d, "merged"), torch_dtype=torch.float32)
    with torch.no_grad():
        hf = m(ids[None], labels=ids[None]).loss.item()
    assert abs(loss - hf) / hf < 1e-3, (loss, hf)
    assert all(math.isfinite(float(x.norm())) for gl in grads for x in gl)


def _family_equivalence(family, extra):
    """LoRA on q,k,v,o through the functional block vs the merged model in transformers (CPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_lora import LoRAConfig, LoRATrainer, TrainConfig, export
    d = tempfile.mkdtemp()
    cfg = AutoConfig.for_model(family, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                               num_key_value_heads=2, num_hidden_layers=2, vocab_size=1000,
                               tie_word_embeddings=False, **extra)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).to(BF).save_pretrained(os.path.join(d, "base"), safe_serialization=True)
    tr = LoRATrainer(os.path.join(d, "base"), CPUBackend(), LoRAConfig(r=4, alpha=8, targets=("q", "k", "v", "o")),
                     TrainConfig(seq_len=32), log=lambda s: None)
    tr.setup()
    g = torch.Generator().manual_seed(1)
    for per in tr.master:
        for b in per[1::2]:
            b.detach().normal_(0, 0.05, generator=g)
    tr._upload_adapters()
    ids = torch.randint(0, 1000, (32,), generator=g)
    lab = torch.roll(ids, -1)
    lab[-1] = -1
    loss, grads, _ = tr.forward_backward(ids, lab)
    export.save_peft(tr, os.path.join(d, "ad"))
    export.save_merged(os.path.join(d, "base"), os.path.join(d, "ad"), os.path.join(d, "merged"))
    m = AutoModelForCausalLM.from_pretrained(os.path.join(d, "merged"), dtype=torch.float32)
    with torch.no_grad():
        hf = m(ids[None], labels=ids[None]).loss.item()
    assert abs(loss - hf) / hf < 1e-3, (family, loss, hf)
    assert all(math.isfinite(float(x.norm())) for gl in grads for x in gl)


def test_qwen2_equivalence():          # q/k/v biases
    _family_equivalence("qwen2", dict(use_sliding_window=False))


def test_qwen3_equivalence():          # per-head q/k norm, head_dim with nh * head_dim != hidden
    _family_equivalence("qwen3", dict(head_dim=32, use_sliding_window=False))


def test_gemma3_equivalence():         # (1+w) sandwich norms, GELU-tanh, attention scale, sliding + full RoPE bases
    _family_equivalence("gemma3_text", dict(head_dim=16, query_pre_attn_scalar=24, sliding_window=64,
                                            layer_types=["sliding_attention", "full_attention"],
                                            rope_local_base_freq=10000.0))


def test_mistral_equivalence():
    _family_equivalence("mistral", dict(sliding_window=None))


def _moe_equivalence(family, extra):
    """MoE checkpoint: attention through the functional block, norm + MoE block on the host (HF modules).
    Loss vs the merged model in transformers, and LoRA gradients vs autograd through it: with W' = W + s·B·A,
    dL/dB = s·(dL/dW')·Aᵀ and dL/dA = s·Bᵀ·(dL/dW'). initializer_range=0.2 makes the MoE block matter: with the
    block's Jacobian dropped the gradients are off by ~0.9 (relative), with it by ~0.03 (bf16)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_lora import LoRAConfig, LoRATrainer, TrainConfig, export
    d = tempfile.mkdtemp()
    cfg = AutoConfig.for_model(family, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                               num_key_value_heads=2, num_hidden_layers=2, vocab_size=1000,
                               tie_word_embeddings=False, initializer_range=0.2, **extra)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).to(BF).save_pretrained(os.path.join(d, "base"), safe_serialization=True)
    lc = LoRAConfig(r=4, alpha=8, targets=("q", "v"))
    tr = LoRATrainer(os.path.join(d, "base"), CPUBackend(), lc, TrainConfig(seq_len=32), log=lambda s: None)
    assert tr.c.moe
    tr.setup()
    g = torch.Generator().manual_seed(1)
    for per in tr.master:
        for b in per[1::2]:
            b.detach().normal_(0, 0.05, generator=g)
    tr._upload_adapters()
    ids = torch.randint(0, 1000, (32,), generator=g)
    lab = torch.roll(ids, -1)
    lab[-1] = -1
    loss, grads, _ = tr.forward_backward(ids, lab)
    export.save_peft(tr, os.path.join(d, "ad"))
    export.save_merged(os.path.join(d, "base"), os.path.join(d, "ad"), os.path.join(d, "merged"))
    m = AutoModelForCausalLM.from_pretrained(os.path.join(d, "merged"), dtype=torch.float32)
    hf = m(ids[None], labels=ids[None]).loss
    hf.backward()
    assert abs(loss - hf.item()) / hf.item() < 1e-3, (family, loss, hf.item())
    names = {"q": "q_proj", "v": "v_proj"}
    for i, per in enumerate(tr.master):
        for j, t in enumerate(lc.targets):
            A, B = per[2 * j].detach(), per[2 * j + 1].detach()
            gw = getattr(m.model.layers[i].self_attn, names[t]).weight.grad
            ref_a, ref_b = lc.scale * B.t() @ gw, lc.scale * gw @ A.t()
            for got, ref, what in ((grads[i][2 * j], ref_a, "A"), (grads[i][2 * j + 1], ref_b, "B")):
                rel = float((got - ref).norm() / (ref.norm() + 1e-12))
                assert rel < 0.1, (family, i, t, what, rel)


def test_qwen3_moe_equivalence():      # softmax top-k router, normalized top-k weights
    _moe_equivalence("qwen3_moe", dict(head_dim=32, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
                                       norm_topk_prob=True, use_sliding_window=False))


def test_qwen2_moe_equivalence():      # q/k/v biases, shared expert with a sigmoid gate
    _moe_equivalence("qwen2_moe", dict(moe_intermediate_size=32, shared_expert_intermediate_size=64, num_experts=8,
                                       num_experts_per_tok=2, use_sliding_window=False))


def test_mixtral_equivalence():
    _moe_equivalence("mixtral", dict(num_local_experts=4, num_experts_per_tok=2, sliding_window=None))


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
