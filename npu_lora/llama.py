"""Llama 계열 디코더를 순수 함수로 — 가중치와 LoRA 어댑터가 모두 인자다.

AOT 컴파일 NPU 는 가중치를 상수로 굽는 걸 전제하지만, 여기서는 가중치를 그래프 입력으로 넘긴다.
그래서 레이어 하나의 그래프를 한 번만 컴파일하고 모든 레이어에 다른 가중치로 재사용할 수 있다.
"""
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BF = torch.bfloat16

# 레이어 그래프에 넘기는 고정 가중치의 순서. HF 키 이름은 load_layer 가 맞춘다.
LAYER_WEIGHTS = ("ln1", "ln2", "wq", "wk", "wv", "wo", "wg", "wu", "wd")
#: optional per-layer tensors of other families: q/k/v biases (Qwen2), per-head q/k RMSNorm weights (Qwen3)
OPTIONAL_WEIGHTS = ("bq", "bk", "bv", "qn", "kn", "ln3", "ln4")
#: Gemma norms are x_norm * (1 + w): loaded as (1 + w) so the shared rms() applies
_NORMS = ("ln1", "ln2", "ln3", "ln4", "qn", "kn")
HF_NAMES = {"bq": "self_attn.q_proj.@bias", "bk": "self_attn.k_proj.@bias", "bv": "self_attn.v_proj.@bias",
            "qn": "self_attn.q_norm", "kn": "self_attn.k_norm",
            "ln3": "pre_feedforward_layernorm", "ln4": "post_feedforward_layernorm",
            "ln1": "input_layernorm", "ln2": "post_attention_layernorm",
            "wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj",
            "wo": "self_attn.o_proj", "wg": "mlp.gate_proj", "wu": "mlp.up_proj", "wd": "mlp.down_proj"}

# LoRA 를 붙일 수 있는 투영과 (입력 차원, 출력 차원) 을 정하는 함수
LORA_TARGETS = {
    "q": lambda c: (c.H, c.q_dim),
    "k": lambda c: (c.H, c.kv_dim),
    "v": lambda c: (c.H, c.kv_dim),
    "o": lambda c: (c.q_dim, c.H),
}


def hf_name(k):
    """HF tensor name of layer weight k (".weight" unless the name says otherwise)."""
    n = HF_NAMES[k]
    return n.replace(".@", ".") if ".@" in n else n + ".weight"


#: the attention half of a layer, for MoE configs (their MLP runs on the host: see `trainer.HostMLP`)
ATTN_WEIGHTS = ("ln1", "wq", "wk", "wv", "wo")
#: model_type -> attention block family of MoE checkpoints whose MLP half runs on the host
MOE_FAMILIES = {"qwen3_moe": "qwen3", "qwen2_moe": "qwen2", "mixtral": "mistral"}


def layer_weights(c):
    """the per-layer tensors this config's layers take, in graph-input order"""
    extra = (("bq", "bk", "bv") if c.qkv_bias else ()) + (("qn", "kn") if c.qk_norm else ()) \
        + (("ln3", "ln4") if c.gemma else ())
    return (ATTN_WEIGHTS if c.moe else LAYER_WEIGHTS) + extra


@dataclass(frozen=True)
class LlamaConfig:
    H: int
    I: int
    nh: int
    nkv: int
    n_layers: int
    vocab: int
    rope_theta: float = 500000.0
    eps: float = 1e-5
    # llama3 rope_scaling: (factor, low_freq_factor, high_freq_factor, original_max_position) 또는 None
    rope_scaling: tuple = None
    family: str = "llama"          # llama | mistral | qwen2 | qwen3 (same block, the flags below differ)
    head_dim: int = 0              # 0: H // nh (Qwen3 sets it explicitly: nh * head_dim may differ from H)
    qkv_bias: bool = False         # Qwen2
    qk_norm: bool = False          # Qwen3 / Gemma 3: per-head RMSNorm on q and k before RoPE
    gemma: bool = False            # Gemma 3: (1+w) norms, sandwich norms around attention and MLP, GELU-tanh,
    attn_scale: float = 0.0        #   query_pre_attn_scalar ** -0.5 (0: 1/sqrt(hd)), sqrt(H)-scaled input embeddings,
    layer_types: tuple = ()        #   sliding layers with their own RoPE base (rope_local_theta)
    rope_local_theta: float = 10000.0
    rope_linear: float = 1.0       # linear RoPE scaling factor of the full-attention layers (Gemma 3 4B+)
    window: int = 0                # sliding window (the sequence length must fit in it)
    moe: bool = False              # MoE checkpoint: the compiled layer is attention + residual only; the
                                   #   post-attention norm and the MoE block run on the host (frozen)

    @property
    def hd(self):
        return self.head_dim or self.H // self.nh

    @property
    def q_dim(self):
        return self.nh * self.hd

    @property
    def kv_dim(self):
        return self.nkv * self.hd

    @property
    def key(self):
        """compiled-graph cache key part for the block variant (empty for plain Llama: old caches stay valid)"""
        if self.family == "llama" and not (self.head_dim or self.qkv_bias or self.qk_norm or self.moe):
            return ""
        return f"-{self.family}-hd{self.hd}-b{int(self.qkv_bias)}-n{int(self.qk_norm)}" + ("-attn" if self.moe else "")

    @classmethod
    def _gemma3(cls, c):
        return _gemma3_from(cls, c)

    @classmethod
    def from_hf(cls, path):
        c = json.load(open(os.path.join(path, "config.json")))
        family = c.get("model_type", "llama")
        if family == "gemma3_text":
            return cls._gemma3(c)
        moe = family in MOE_FAMILIES
        if moe:
            family = MOE_FAMILIES[family]
        if family not in ("llama", "mistral", "qwen2", "qwen3"):
            raise NotImplementedError(f"model_type {family!r}: npu-lora implements llama, mistral, qwen2, qwen3, "
                                      f"gemma3_text and the MoE checkpoints {sorted(MOE_FAMILIES)}")
        if c.get("use_sliding_window") or (family == "mistral" and c.get("sliding_window")):
            raise NotImplementedError("sliding-window attention is not implemented")
        rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
        kind = rs.get("rope_type", rs.get("type"))
        if kind not in (None, "default", "llama3"):
            raise NotImplementedError(f"rope_scaling {kind!r} is not supported yet")
        scaling = ((float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"]),
                    int(rs["original_max_position_embeddings"])) if kind == "llama3" else None)
        return cls(H=c["hidden_size"], I=c["intermediate_size"], nh=c["num_attention_heads"],
                   nkv=c.get("num_key_value_heads", c["num_attention_heads"]),
                   n_layers=c["num_hidden_layers"], vocab=c["vocab_size"],
                   rope_theta=float(c.get("rope_theta", rs.get("rope_theta", 10000.0))),
                   eps=float(c.get("rms_norm_eps", 1e-5)), rope_scaling=scaling, family=family,
                   head_dim=int(c.get("head_dim") or 0) if family == "qwen3" else 0,
                   qkv_bias=family == "qwen2" or bool(c.get("attention_bias", False)), qk_norm=family == "qwen3",
                   moe=moe)


def _gemma3_from(cls, c):
    if c.get("final_logit_softcapping") or c.get("attn_logit_softcapping"):
        raise NotImplementedError("logit soft-capping is not implemented")
    if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise NotImplementedError(f"activation {c.get('hidden_activation')}")
    n = c["num_hidden_layers"]
    kinds = c.get("layer_types") or ["full_attention" if (i + 1) % c.get("sliding_window_pattern", 6) == 0
                                     else "sliding_attention" for i in range(n)]
    rp = c.get("rope_parameters") or {}
    full = rp.get("full_attention", rp) if isinstance(rp, dict) else {}
    local = rp.get("sliding_attention", {}) if isinstance(rp, dict) else {}
    rs = c.get("rope_scaling") or full
    lin = float(rs.get("factor", 1.0)) if rs.get("rope_type", rs.get("type")) == "linear" else 1.0
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return cls(H=c["hidden_size"], I=c["intermediate_size"], nh=c["num_attention_heads"],
               nkv=c.get("num_key_value_heads", c["num_attention_heads"]), n_layers=n, vocab=c["vocab_size"],
               rope_theta=float(c.get("rope_theta", full.get("rope_theta", 1e6))),
               eps=float(c.get("rms_norm_eps", 1e-6)), family="gemma3_text", head_dim=int(hd), qk_norm=True,
               gemma=True, attn_scale=float(c.get("query_pre_attn_scalar", hd)) ** -0.5, layer_types=tuple(kinds),
               rope_local_theta=float(c.get("rope_local_base_freq", local.get("rope_theta", 10000.0))),
               rope_linear=lin, window=int(c.get("sliding_window") or 0))


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def rms(x, g, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * g


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


def proj(x, w, lora, scale):
    """x @ w^T (+ LoRA). lora 는 (A[r,in], B[out,r]) 또는 None."""
    y = x @ w.t()
    if lora is not None:
        a, b = lora
        y = y + (x @ a.t()) @ b.t() * scale
    return y


def rope_mm(x, cos, sin, rot):
    """rope() 와 같은 값. rotate_half(x) = x @ rot — slice/cat 대신 [hd,hd] 고정 행렬곱.

    furiosa 컴파일러는 slice·neg·cat 재배치가 든 그래프에서 컴파일 시간이 폭증했다(어텐션 순전파
    15분 초과). 행렬곱은 NPU 가 가장 잘하는 연산이고 hd=128 이라 비용도 작다.
    """
    return x * cos + (x @ rot) * sin


def rotate_half_matrix(hd, dtype=BF):
    """x @ R == cat(-x[..., hd/2:], x[..., :hd/2])"""
    r = torch.zeros(hd, hd)
    h = hd // 2
    r[h:, :h] = -torch.eye(h)
    r[:h, h:] = torch.eye(h)
    return r.to(dtype)


def layer(c, h, w, lora, scale, cos, sin, mask, rot):
    """디코더 레이어 하나. w: LAYER_WEIGHTS 순서의 dict, lora: {"q": (A,B), ...}.

    GQA 는 K/V 를 repeat_interleave 로 복제하지 않고, q 를 [kv헤드, 반복×S, hd] 로 묶어 그룹당
    bmm 한 번으로 계산한다 — 같은 값이고, 복제(expand+clone)가 컴파일러 레이아웃 탐색을 키우지 않는다.
    q 헤드 h 가 kv 헤드 h // rep 을 쓰는 HF 규칙과 같은 순서다.
    """
    S = h.shape[0]
    rep = c.nh // c.nkv
    x = rms(h, w["ln1"], c.eps)
    # RoPE 는 전치하기 전 [S, heads, hd] 배치에서 건다. 전치한 q 에 걸면 원소별 연산의 두 피연산자
    # 배치가 달라져([nkv,rep,S,hd] vs [S,nkv,rep,hd]) furiosa 가 "verification of operator failed" 로 멈춘다.
    cs, sn = cos.unsqueeze(1), sin.unsqueeze(1)                          # [S, 1, hd]
    q, k, v = proj(x, w["wq"], lora.get("q"), scale), proj(x, w["wk"], lora.get("k"), scale), \
        proj(x, w["wv"], lora.get("v"), scale)
    if c.qkv_bias:
        # Qwen2's q/k biases are large next to x @ w (|q|, |k| reach the hundreds), so bf16 q + bq and q . k
        # keep only ~2-3 significant digits of the attention logits: on the CPU alone the bf16 LoRA gradients
        # were 54% (q) / 21% (v) off an fp32 reference, and RNGD vs CPU differed by as much (#33). q and k from
        # the bias add through RoPE and q . k^T stay fp32 (1.6% / 1%); the projections themselves remain bf16.
        q, k, v = q.float() + w["bq"].float(), k.float() + w["bk"].float(), v + w["bv"]
        cs, sn, rot = cs.float(), sn.float(), rot.float()
    q, k = q.view(S, c.nh, c.hd), k.view(S, c.nkv, c.hd)
    if c.qk_norm:                                                        # per head, over hd
        q, k = rms(q, w["qn"], c.eps), rms(k, w["kn"], c.eps)
    q = rope_mm(q, cs, sn, rot).transpose(0, 1)
    k = rope_mm(k, cs, sn, rot).transpose(0, 1)
    v = v.view(S, c.nkv, c.hd).transpose(0, 1)
    qg = q.reshape(c.nkv, rep * S, c.hd)                                 # [nkv, rep*S, hd]
    att_scale = c.attn_scale or 1.0 / math.sqrt(c.hd)          # (`scale` is the LoRA scale)
    att = (qg @ k.transpose(-1, -2)).float().view(c.nkv, rep, S, S) * att_scale + mask
    att = torch.softmax(att, -1).to(h.dtype).view(c.nkv, rep * S, S)
    o = (att @ v).view(c.nh, S, c.hd)                                   # [nh, S, hd]
    o = proj(o.transpose(0, 1).reshape(S, c.q_dim), w["wo"], lora.get("o"), scale)
    if c.moe:                                    # the MLP half runs on the host (trainer.HostMLP)
        return h + o
    if c.gemma:                                  # sandwich norms: post-attention, pre/post feed-forward
        h = h + rms(o, w["ln2"], c.eps)
        x = rms(h, w["ln3"], c.eps)
        m = (gelu_tanh(x @ w["wg"].t()) * (x @ w["wu"].t())) @ w["wd"].t()
        return h + rms(m, w["ln4"], c.eps)
    h = h + o
    x = rms(h, w["ln2"], c.eps)
    return h + (F.silu(x @ w["wg"].t()) * (x @ w["wu"].t())) @ w["wd"].t()


def head_loss(c, h, norm, lm, iota, labels, inv_n):
    """다음 토큰 cross-entropy (토큰 평균).

    labels[S,1] 은 정답 토큰 id 를 float 로(학습에서 뺄 위치는 -1), iota[1,V] 는 0..V-1.
    정수 label 의 gather/scatter 역전파를 피하려고 장치에서 비교로 one-hot 을 만든다 —
    [S,V] one-hot 을 매 step host 에서 올리지 않아도 된다(V=128k 면 S=512 에서 128 MB).
    0차원 텐서는 furiosa 컴파일러가 받지 못하므로 loss 는 [1,1] 로 남긴다.
    """
    logits = (rms(h, norm, c.eps) @ lm.t()).float()
    logp = torch.log_softmax(logits, -1)
    onehot = (iota == labels).float()
    return -(logp * onehot).sum(dim=(0, 1), keepdim=True) * inv_n


# ---------------------------------------------------------------- 어휘 조각 cross-entropy
#
# [S, V] logits 를 한 번에 만들면 V=128k 에서 65 MB(fp32, S=128)라 NPU 온칩 메모리를 넘고, furiosa 가
# 그래프를 여러 장치 부분 그래프로 쪼개다 거부한다("multiple internal subgraphs are not supported").
# 어휘를 Vc 조각으로 나눠 조각마다 [S, Vc] 만 만들고, 조각 사이에는 행마다 숫자 몇 개만 오간다.

def chunk_logits(x, lm_j, bias_j):
    """fp32 logits of chunk j. With a bf16 matmul, RNGD gave the target logit t up to 0.06 above the row max m
    of the same graph (t and m read differently rounded logits), so lse - t went negative on confident tokens
    (#33: 11 of 400 steps reported a negative loss). In fp32 the three statistics read the same values."""
    return x.float() @ lm_j.float().t() + bias_j


def ce_chunk_stats(x, lm_j, bias_j, iota_j, labels):
    """조각 j 의 (행 최댓값 m, exp 합 s, 정답 logit t). x = rms(h) [S,H], lm_j [Vc,H].

    bias_j[1,Vc] 는 보통 0, 어휘를 Vc 배수로 맞추려 덧붙인 가짜 토큰 자리에는 -1e9.
    정답이 이 조각에 없으면 t = 0 (iota_j 에 그 id 가 없으니 one-hot 이 비어 있다).
    """
    logits = chunk_logits(x, lm_j, bias_j)
    m = logits.amax(-1, keepdim=True)
    s = torch.exp(logits - m).sum(-1, keepdim=True)
    t = (logits * (iota_j == labels).float()).sum(-1, keepdim=True)
    return m, s, t


def ce_chunk_grad(x, lm_j, bias_j, iota_j, labels, lse, w, acc):
    """acc + 조각 j 가 x 기울기에 보태는 몫 ((softmax_j - onehot_j) * w) @ lm_j.

    lse[S,1] 은 전 어휘 logsumexp(host 가 조각 통계로 계산), w[S,1] 는 행 가중치(유효 토큰 1/n, 그 외 0).
    누적을 그래프 안에서 해서 장치 텐서에 eager 연산을 하지 않는다.
    """
    logits = chunk_logits(x, lm_j, bias_j)
    p = torch.exp(logits - lse)
    g = (p - (iota_j == labels).float()) * w
    return acc + (g.to(x.dtype) @ lm_j).float()


def ce_combine(stats):
    """[(m, s, t)] (조각별, 각 [S,1]) → (lse[S,1], target_logit[S,1])"""
    m = torch.cat([x[0] for x in stats], -1)
    s = torch.cat([x[1] for x in stats], -1)
    M = m.amax(-1, keepdim=True)
    lse = M + torch.log((s * torch.exp(m - M)).sum(-1, keepdim=True))
    t = torch.cat([x[2] for x in stats], -1).sum(-1, keepdim=True)
    return lse, t


def vocab_chunks(V, target=4096):
    """(조각 수, 조각 크기, 덧붙일 가짜 토큰 수). 나누어떨어지면 덧붙이지 않는다."""
    for vc in range(target, target // 2, -8):
        if V % vc == 0:
            return V // vc, vc, 0
    n = -(-V // target)
    return n, target, n * target - V


def rope_inv_freq(c):
    """RoPE 주파수. Llama 3.x 의 llama3 rope_scaling(저주파 성분을 factor 로 나누고 중간 대역은 보간)을
    HF `_compute_llama3_parameters` 와 같게 적용한다 — 빼먹으면 사전학습 때와 위치 인코딩이 달라진다."""
    inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.hd, 2, dtype=torch.float32) / c.hd))
    if not c.rope_scaling:
        return inv
    factor, lo, hi, old = c.rope_scaling
    lo_wav, hi_wav = old / lo, old / hi
    wav = 2 * math.pi / inv
    out = torch.where(wav > lo_wav, inv / factor, inv)
    smooth = (old / wav - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = ~(wav < hi_wav) & ~(wav > lo_wav)
    return torch.where(medium, smoothed, out)


def rope_tables(c, S, dtype=BF, kind="full_attention"):
    pos = torch.arange(S, dtype=torch.float32)
    if c.gemma:
        theta = c.rope_local_theta if kind == "sliding_attention" else c.rope_theta
        inv = 1.0 / (theta ** (torch.arange(0, c.hd, 2, dtype=torch.float32) / c.hd))
        if kind != "sliding_attention":
            inv = inv / c.rope_linear
    else:
        inv = rope_inv_freq(c)
    ang = torch.outer(pos, inv)
    ang = torch.cat((ang, ang), -1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def causal_mask(S, doc_ids=None):
    """[S,S] 가산 마스크. doc_ids 를 주면 문서 경계를 넘지 않는 블록 대각 인과 마스크."""
    m = torch.full((S, S), -1e9).triu(1)
    if doc_ids is not None:
        d = torch.as_tensor(doc_ids)
        m = m.masked_fill(d[:, None] != d[None, :], -1e9)
    return m


# ---------------------------------------------------------------- 가중치 읽기 (safetensors)

class Weights:
    """HF safetensors 디렉터리에서 필요한 텐서만 읽는다(8B 를 통째로 올리지 않는다)."""

    def __init__(self, path):
        from safetensors import safe_open
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.where = json.load(open(idx))["weight_map"]
        else:
            self.where = {}
            with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
                for k in f.keys():
                    self.where[k] = "model.safetensors"
        self._open = {}

    def get(self, name, dtype=BF):
        from safetensors import safe_open
        fn = self.where[name]
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return self._open[fn].get_tensor(name).to(dtype)

    def shape(self, name):
        """the tensor's shape from the safetensors header (reads no data)"""
        from safetensors import safe_open
        fn = self.where[name]
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return tuple(self._open[fn].get_slice(name).get_shape())

    def layer(self, i, c=None):
        keys = layer_weights(c) if c is not None else LAYER_WEIGHTS
        out = {}
        for k in keys:
            if c is not None and c.gemma and k in _NORMS:
                out[k] = (1.0 + self.get(f"model.layers.{i}.{hf_name(k)}", torch.float32)).to(BF)
            else:
                out[k] = self.get(f"model.layers.{i}.{hf_name(k)}")
        return out

    def embed(self):
        return self.get("model.embed_tokens.weight")

    def norm(self, c=None):
        if c is not None and c.gemma:
            return (1.0 + self.get("model.norm.weight", torch.float32)).to(BF)
        return self.get("model.norm.weight")

    def lm_head(self):
        return self.get("lm_head.weight") if "lm_head.weight" in self.where else self.embed()


def resolve_model_path(name_or_path):
    """로컬 경로가 아니면 HF 캐시의 스냅샷 경로를 찾는다(오프라인)."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
