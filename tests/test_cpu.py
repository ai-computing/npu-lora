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
from npu_train import graph as G, llama as L  # noqa: E402
from npu_train.backends import CPUBackend  # noqa: E402

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
    from npu_train import LoRATrainer, LoRAConfig, TrainConfig
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


def test_two_device_split_and_resident_adamw_equal_one_device():
    """full FT with the layers split over two backends, and with the first layer of each device's range
    updated by the resident device AdamW graph, gives the same weights as the one-device offload pipeline."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    from npu_train import LoRATrainer, LoRAConfig, TrainConfig
    d = tempfile.mkdtemp()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=4, vocab_size=1000, rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(BF).save_pretrained(d, safe_serialization=True)
    g = torch.Generator().manual_seed(3)
    batches = []
    for _ in range(3):
        mb = []
        for _ in range(2):
            ids = torch.randint(0, 1000, (32,), generator=g)
            lab = torch.roll(ids, -1)
            lab[-1] = -1
            mb.append((ids, lab, None))
        batches.append(mb)

    def run(backends, resident):
        tc = TrainConfig(seq_len=32, full=True, accum=2, lr=1e-3, warmup=1, steps=4, grad_clip=0.5,
                         resident_layers=resident)
        tr = LoRATrainer(d, backends, LoRAConfig(), tc, log=lambda s: None)
        tr.setup()
        losses = [tr.train_step(b)["loss"] for b in batches]
        tr.sync_params()
        return tr, losses

    one, l1 = run(CPUBackend(), 0)
    two, l2 = run([CPUBackend(), CPUBackend()], 0)
    res, l3 = run([CPUBackend(), CPUBackend()], 1)
    assert two._dev == [0, 0, 1, 1] and res._resident == {0, 2}
    for other, losses in ((two, l2), (res, l3)):
        assert max(abs(a - b) for a, b in zip(l1, losses)) < 1e-3, (l1, losses)
        for a, b in zip(one.params, other.params):
            assert _rel(a.detach(), b.detach()) < 1e-2, _rel(a.detach(), b.detach())
    # updates (w - w0), not weights, so a wrong device update cannot hide behind the weights' size
    w0 = LoRATrainer(d, CPUBackend(), LoRAConfig(), TrainConfig(full=True), log=lambda s: None).W
    for layer in (0, 2):                         # the resident layers
        base = w0.get(f"model.layers.{layer}.self_attn.q_proj.weight", torch.float32)
        du, dr = one.master[layer][2].detach() - base, res.master[layer][2].detach() - base
        assert float(dr.abs().max()) > 0 and _rel(dr, du) < 5e-2, _rel(dr, du)


def test_resident_shapes_from_header():
    """the resident AdamW graph shapes come from the safetensors header alone (compile --adamw-only)"""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    from npu_train.optim_device import resident_shapes
    d = tempfile.mkdtemp()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=1, vocab_size=1000, tie_word_embeddings=False)
    LlamaForCausalLM(cfg).to(BF).save_pretrained(d, safe_serialization=True)
    W = L.Weights(d)
    assert resident_shapes(W, L.LlamaConfig.from_hf(d)) == [(32, 64), (64, 64), (64, 128), (128, 64)]
    assert W.shape("model.layers.0.mlp.down_proj.weight") == tuple(W.get("model.layers.0.mlp.down_proj.weight").shape)


def test_visible_chip_picks_the_nth_assigned_card():
    """a two-card pod on a node with npu0, npu1, npu3 assigned npu1,npu3 -> ordinals 1 and 2"""
    from npu_train import backends
    real, env = backends.os.listdir, dict(os.environ)
    backends.os.listdir = lambda d: ["npu0", "npu0pe0", "npu1", "npu1mgmt", "npu3", "npu3pe0-1"]
    try:
        os.environ["NPU_LORA_RNGD"] = "npu1,npu3"
        assert [backends.visible_chip(0), backends.visible_chip(1)] == [1, 2]
        os.environ["NPU_LORA_RNGD"] = "npu3"
        assert backends.visible_chip() == 2
    finally:
        backends.os.listdir = real
        os.environ.clear()
        os.environ.update(env)


def test_serialized_graph_call_holds_the_device_alloc_lock():
    """#40: a compiled graph call must not overlap a device allocation from a layer worker thread"""
    import threading
    from npu_train import backends

    class Mod:
        tag = "edf"

        def __call__(self, x):
            assert backends.DEVICE_ALLOC_LOCK.locked()
            return x + 1

    m = backends.Serialized(Mod())
    assert m(1) == 2 and m.tag == "edf" and not backends.DEVICE_ALLOC_LOCK.locked()
    order = []
    with backends.DEVICE_ALLOC_LOCK:               # a worker is allocating: the call waits for it
        t = threading.Thread(target=lambda: order.append(m(0)))
        t.start()
        t.join(0.2)
        order.append("alloc done")
    t.join()
    assert order == ["alloc done", 1]


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


def test_fold_neg_add_keeps_silu_backward():
    # the SiLU backward after scalarize_constants has neg + add.Scalar (1 - sigmoid): RBLN gets rsub instead
    x, g = torch.randn(8, 16), torch.randn(8, 16)
    fn = lambda x_, g_: torch.func.vjp(torch.nn.functional.silu, x_)[1](g_)[0]
    gm = G.build(fn, [x, g])
    ref = gm(x, g)
    assert G.fold_neg_add(gm) >= 1
    ops = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    assert "aten.neg.default" not in ops and "aten.rsub.Scalar" in ops
    assert torch.allclose(gm(x, g), ref, atol=1e-6)


def test_rbln_host_parts_place_groups_on_the_host(monkeypatch):
    import types
    from npu_train.backends import RBLNBackend
    monkeypatch.setitem(sys.modules, "rebel", types.SimpleNamespace(__version__="test"))
    assert RBLNBackend.group("llama-v2-H64-S32-r4-qv-s2.0-fwd-p3of15-ab12cd") == "fwd"
    assert RBLNBackend.group("llama-v3-H64-S32-Vc1000-rms-bwd-p0of1-ab12cd") == "rms-bwd"
    assert RBLNBackend.group("sqnorm-64x64") is None
    be = RBLNBackend(0, cache_dir=tempfile.mkdtemp())          # default: the layer forward on the host
    assert be.host_parts == ("fwd",)
    gm = G.build(lambda x: x * 2 + 1, [torch.randn(4, 4)])
    fn = be.compile("k-fwd-p0of1-abc", gm, [torch.randn(4, 4)])  # placed on the host: no compiler call
    x = torch.randn(4, 4)
    assert torch.equal(fn(x), x * 2 + 1) and be.placed == ["k-fwd-p0of1-abc"] and be.seconds["host"] > 0
    assert RBLNBackend(0, cache_dir=tempfile.mkdtemp(), host_parts=()).host_parts == ()


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
    stats = [L.ce_chunk_stats(x, *c[:2]) for c in chunks]
    lse = L.ce_combine(stats)
    t = L.target_logits(x, lm, labels)
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
    from npu_train import LoRAConfig, LoRATrainer, TrainConfig, export
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
    ref_loss, ref = tr.reference_grads(ids, lab)          # the fp64 reference `check` uses
    assert abs(ref_loss - hf) / hf < 1e-3, (ref_loss, hf)
    assert all(_rel(x, r) < 5e-2 for gl, rl in zip(grads, ref) for x, r in zip(gl, rl))


def _family_equivalence(family, extra):
    """LoRA on q,k,v,o through the functional block vs the merged model in transformers (CPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_train import LoRAConfig, LoRATrainer, TrainConfig, export
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


def test_qwen2_large_bias_lora_grads_match_fp32():
    """Qwen2-sized q/k biases (|k| ~ 100s) must not cost the bf16 layer graph its LoRA gradients (#33):
    the compiled-graph form of the backward vs fp32 autograd on the same inputs."""
    c = L.LlamaConfig(H=64, I=128, nh=4, nkv=2, n_layers=1, vocab=1000, family="qwen2", qkv_bias=True)
    S = 64
    args, g_out = _layer_inputs(S)
    args = [*args, g_out]
    gen = torch.Generator().manual_seed(3)
    nw = len(L.LAYER_WEIGHTS)
    # q, k, v biases go after the 9 Llama weights (L.layer_weights order); k's are large like Qwen2.5's
    bias = [(torch.randn(C.H, generator=gen) * 8).to(BF), (torch.randn(C.kv_dim, generator=gen) * 60).to(BF),
            (torch.randn(C.kv_dim, generator=gen) * 0.5).to(BF)]
    args = args[:1 + nw] + bias + args[1 + nw:]
    bwd = G.layer_bwd_fn(c, ("q", "v"), 2.0)
    got = G.build(bwd, args)(*args)
    ref = bwd(*[a.float() for a in args])
    for i in range(1, 5):                                    # A_q, B_q, A_v, B_v
        assert _rel(got[i], ref[i]) < 5e-2, (i, _rel(got[i], ref[i]))


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
    from npu_train import LoRAConfig, LoRATrainer, TrainConfig, export
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
    # The host MoE block runs in bf16 against an fp32 reference. Measured loss gaps: mixtral 1.0e-4; qwen2_moe
    # (shared expert) 8.3e-4 on transformers 5.1 and 1.03e-3 on 5.20; the dense families stay below 3.1e-5.
    assert abs(loss - hf.item()) / hf.item() < 2e-3, (family, loss, hf.item())
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


def test_moe_full_finetuning():
    """MoE full fine-tuning: attention on the device path, norm + router + experts trained on the host.
    The host part's gradients equal HF autograd; loss decreases; the export loads in HF with the trainer's loss."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_train import LoRAConfig, LoRATrainer, TrainConfig, export
    from npu_train import moe as M
    d = tempfile.mkdtemp()
    cfg = AutoConfig.for_model("qwen3_moe", hidden_size=64, intermediate_size=128, num_attention_heads=4,
                               num_key_value_heads=2, num_hidden_layers=2, vocab_size=1000, head_dim=32,
                               moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
                               use_sliding_window=False, tie_word_embeddings=False, initializer_range=0.2)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).to(BF).save_pretrained(os.path.join(d, "base"), safe_serialization=True)
    tc = TrainConfig(seq_len=32, full=True, lr=2e-3, warmup=0, steps=8, grad_clip=0.0)
    tr = LoRATrainer(os.path.join(d, "base"), CPUBackend(), LoRAConfig(), tc, log=lambda s: None)
    assert tr.c.moe
    tr.setup()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (32,), generator=g)
    lab = torch.roll(ids, -1)
    lab[-1] = -1
    # 1) host-part gradients of the first step vs HF autograd on the same (still untrained) weights
    seen = {}
    step = M.HostMLP.step

    def capture(self, clip, lr, k):
        seen.update({(i, n): x.clone() / k for i, per in enumerate(self.acc) if per is not None
                     for (n, _), x in zip(self.named_parameters(i), per)})
        return step(self, clip, lr, k)
    M.HostMLP.step = capture
    try:
        first = tr.train_step([(ids, lab, None)])["loss"]
    finally:
        M.HostMLP.step = step
    ref = AutoModelForCausalLM.from_pretrained(os.path.join(d, "base"), dtype=torch.float32)
    loss = ref(ids[None], labels=ids[None]).loss
    loss.backward()
    assert abs(first - loss.item()) / loss.item() < 1e-2
    hf = dict(ref.named_parameters())
    checked = 0
    for (i, name), got in seen.items():
        want = hf[name].grad
        if want is None or float(want.norm()) == 0:
            continue
        rel = float((got - want).norm() / want.norm())
        assert rel < 0.1, (name, rel)
        checked += 1
    assert checked >= 6, checked                 # norm, router, experts of both layers
    # 2) training reduces the loss
    losses = [tr.train_step([(ids, lab, None)])["loss"] for _ in range(6)]
    assert losses[-1] < first * 0.8, (first, losses)
    # 3) the export is a full HF model with the trained weights
    tr.sync_params()
    with torch.no_grad():
        now, _, _ = tr.forward_backward(ids, lab)
    out = export.save_full(tr, os.path.join(d, "full"))
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    with torch.no_grad():
        hf_now = m(ids[None], labels=ids[None]).loss.item()
    assert abs(now - hf_now) / hf_now < 2e-2, (now, hf_now)


def test_host_mlp_keep_graph_equals_recompute():
    """HostMLP.keep_graph reuses the forward's autograd graph in backward: same input gradients (LoRA) and the
    same host-part parameter gradients (full fine-tuning) as recomputing the forward."""
    from npu_train import moe as M
    d = tempfile.mkdtemp()
    _tiny_moe(d)
    g = torch.Generator().manual_seed(2)
    a = (torch.randn(32, 64, generator=g)).to(BF)
    gy = (torch.randn(32, 64, generator=g) * 0.1).to(BF)
    for trainable in (False, True):
        out = []
        for keep in (False, True):
            h = M.HostMLP(d, trainable=trainable, keep_graph=keep)
            y = h.forward(0, a)
            gx = h.backward(0, a, gy)
            out.append((y, gx, [x.clone() for x in h.acc[0]] if trainable else []))
        (y0, g0, p0), (y1, g1, p1) = out
        assert torch.equal(y0, y1) and torch.equal(g0, g1)
        assert all(torch.equal(u, v) for u, v in zip(p0, p1))


def test_hot_expert_packing():
    """pack(): rows grouped by hot expert, each at slot * nb + rank; nb = bucket of the busiest expert; the batched
    expert graph equals the per-expert host computation on the packed rows"""
    from npu_train import moe_device as MD
    slot = torch.tensor([2, 0, 2, 1, 2, 0])
    order, place, nb = MD.pack(slot, 3)
    assert nb == 8 and slot[order].tolist() == [0, 0, 1, 2, 2, 2]
    assert place.tolist() == [0, 1, 8, 16, 17, 18]
    g = torch.Generator().manual_seed(0)
    x, gu, dn = torch.randn(3, nb, 16, generator=g), torch.randn(3, 24, 16, generator=g), torch.randn(3, 16, 12, generator=g)
    y = MD.experts_fwd(x, gu, dn)
    for e in range(3):
        assert torch.allclose(y[e], MD.expert_fwd(x[e], gu[e], dn[e]), atol=1e-5)


def _tiny_moe(d):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("qwen3_moe", hidden_size=64, intermediate_size=128, num_attention_heads=4,
                               num_key_value_heads=2, num_hidden_layers=2, vocab_size=1000, head_dim=32,
                               moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
                               use_sliding_window=False, tie_word_embeddings=False, initializer_range=0.2)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).to(BF).save_pretrained(d, safe_serialization=True)


def test_moe_hot_experts_on_device_match_host():
    """hot experts through the device expert graphs (CPU backend) == every expert on the host: LoRA adapter
    gradients and full fine-tuning host-part gradients (incl. the stacked expert weights) with half of each
    layer's experts forced onto the device path."""
    from npu_train import LoRAConfig, LoRATrainer, TrainConfig
    from npu_train import moe as M
    d = tempfile.mkdtemp()
    _tiny_moe(d)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (32,), generator=g)
    lab = torch.roll(ids, -1)
    lab[-1] = -1

    def make(full, frac):
        tc = TrainConfig(seq_len=32, full=full, lr=2e-3, warmup=0, steps=4, grad_clip=0.0, moe_hot_fraction=frac)
        tr = LoRATrainer(d, CPUBackend(), LoRAConfig(r=4, alpha=8, targets=("q", "v")), tc, log=lambda s: None)
        tr.setup()
        for w in tr.moe_wraps:
            if w is not None:
                w.hot = set(range(0, w.num_experts, 2))           # every other expert on the device
        return tr

    def rel(a, b):
        return float((a - b).norm() / (b.norm() + 1e-12))
    # LoRA: adapter gradients (they flow through the experts' input gradients)
    host, hyb = make(False, 0.0), make(False, 0.5)
    for per in (*host.master, *hyb.master):                    # non-zero B so every adapter gets a gradient
        for b in per[1::2]:
            b.detach().fill_(0.05)
    host._upload_adapters(), hyb._upload_adapters()
    _, gh, _ = host.forward_backward(ids, lab)
    _, gd, _ = hyb.forward_backward(ids, lab)
    # batched, per layer (2 layers): forward and backward (HostMLP keeps the forward's graph, no recompute) —
    # not one call per hot expert
    assert hyb.moe_stats()["device_calls"] == 4, hyb.moe_stats()
    for a, b in zip((x for gl in gd for x in gl), (x for gl in gh for x in gl)):
        assert rel(a, b) < 5e-2, rel(a, b)                     # bf16: host autograd vs compiled vjp
    # full fine-tuning: host-part gradients at the first step
    seen = {}
    step = M.HostMLP.step

    def capture(self, clip, lr, k):
        seen[id(self)] = {n: x.clone() for i, per in enumerate(self.acc) if per is not None
                          for (n, _), x in zip(self.named_parameters(i), per)}
        return step(self, clip, lr, k)
    M.HostMLP.step = capture
    try:
        host, hyb = make(True, 0.0), make(True, 0.5)
        host.train_step([(ids, lab, None)])
        hyb.train_step([(ids, lab, None)])
    finally:
        M.HostMLP.step = step
    a, b = seen[id(hyb.host_mlp)], seen[id(host.host_mlp)]
    assert any("experts.gate_up_proj" in n for n in a)
    for n in b:
        assert rel(a[n], b[n]) < 5e-2, (n, rel(a[n], b[n]))


def test_routing_pins_record_and_force_experts():
    """npu-lora check on MoE: the reference run can be pinned to another run's experts (moe.RoutingPins)"""
    from npu_train.moe import HostMLP, RoutingPins
    d = tempfile.mkdtemp()
    _tiny_moe(d)
    mlp = HostMLP(d)
    a = torch.randn(16, mlp.blocks[0][0].weight.shape[0], dtype=torch.bfloat16)
    rec = RoutingPins(mlp.blocks)
    rec.mode = "record"
    with rec:
        y0 = mlp.forward(0, a)
    assert 0 in rec.ids and rec.ids[0].shape[0] == 16
    other = RoutingPins(mlp.blocks)
    other.ids = {0: torch.flip(rec.ids[0], dims=[0])}            # different experts per token
    other.mode = "pin"
    with other:
        y1 = mlp.forward(0, a)
    assert not torch.equal(y0, y1)
    same = RoutingPins(mlp.blocks)
    same.ids, same.mode = rec.ids, "pin"
    with same:
        y2 = mlp.forward(0, a)
    torch.testing.assert_close(y2, y0)                           # pinned to its own choice: unchanged output
    flipped = RoutingPins(mlp.blocks)
    flipped.ids = other.ids
    assert rec.flips(flipped)[0] > 0

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
