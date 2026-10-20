"""LoRA·전체 파인튜닝 학습 루프.

장치(NPU)가 하는 일은 세 그래프뿐이다 — 레이어 순전파, 레이어 역전파(순전파 재계산 포함), head
(loss 와 hidden 기울기). 셋 다 한 번만 컴파일하고 모든 레이어·step 에 재사용한다.
어댑터는 host 에 fp32 master 로 두고 AdamW 로 갱신한 뒤, step 마다 bf16 으로 장치에 올린다
(Llama-8B, r=16, q·v 기준 13.6 MB — 갱신 비용은 작다).

전체 파인튜닝은 CPU offload 파이프라인으로 돈다(_train_step_full): fp32 master·AdamW 상태는 host,
장치에는 bf16 가중치와 장치 누적 기울기만 둔다. 레이어별로 (1) 마지막 micro-batch 역전파가 끝나는 대로
기울기를 다른 스레드에서 내리고, (2) step 끝에 host AdamW·업로드를 레이어 0 부터 순서대로 돌리며,
(3) 다음 step 의 순전파는 해당 레이어가 준비되는 대로 들어간다 — 갱신 결과는 겹치지 않을 때와 같다.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import torch

from . import graph as G
from . import llama as L

BF = L.BF


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    targets: tuple = ("q", "v")

    @property
    def scale(self):
        return self.alpha / self.r


@dataclass
class TrainConfig:
    seq_len: int = 512
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    accum: int = 1                 # gradient accumulation (micro-batch 1 시퀀스씩)
    warmup: int = 10
    steps: int = 100
    seed: int = 0
    log_every: int = 1
    full: bool = False             # True: LoRA 대신 레이어 가중치 9종 전부를 학습(전체 파인튜닝)
    moe_hot_fraction: float = 0.0  # MoE: fraction of each layer's experts computed on the device (most routed)
    moe_replan_every: int = 1      #   re-choose them from routing counts every N steps
    resident_layers: int = 0       # full FT, per device: the first N layers of each device's range keep fp32
                                   #   master + AdamW state on the device and update there (no offload)


class LoRATrainer:
    def __init__(self, model_path, backend, lora=LoRAConfig(), train=TrainConfig(), log=print, max_mm=None):
        self.path = L.resolve_model_path(model_path)
        self.c = L.LlamaConfig.from_hf(self.path)
        # backend may be a list: full fine-tuning then splits the decoder layers into contiguous ranges,
        # one per device (layer i on device i*D//n), and the head runs on the last device.
        self.bes = list(backend) if isinstance(backend, (list, tuple)) else [backend]
        self.be = self.bes[0]
        n_dev = len(self.bes)
        if n_dev > 1 and (not train.full or self.c.moe):
            raise ValueError("several devices are supported for dense full fine-tuning only")
        if train.resident_layers and not train.full:
            raise ValueError("resident_layers needs full fine-tuning")
        self._dev = [i * n_dev // self.c.n_layers for i in range(self.c.n_layers)]
        self.be_head = self.bes[-1]
        first = {d: self._dev.index(d) for d in range(n_dev)}
        # layers whose optimizer state stays on the device: the first N of each device's range, so the next
        # step's forward (which needs layer 0 first) does not wait for their updates
        self._resident = {i for i in range(self.c.n_layers) if i - first[self._dev[i]] < train.resident_layers}
        self.lora = lora
        self.tc = train
        self.log = log
        self.W = L.Weights(self.path)
        self.step_no = 0
        # 그래프 조각 크기(행렬곱 수). furiosa 는 큰 융합 그래프를 컴파일하지 못해 기본 2, 나머지는 통째로.
        self.max_mm = max_mm if max_mm is not None else (2 if self.be.name == "furiosa" else 0)
        self.host_mlp = None          # MoE: post-attention norm + MoE block on the host (moe.HostMLP)
        self.moe_dev, self.moe_wraps = None, []   # MoE: hot experts on the device (moe_device)
        torch.manual_seed(train.seed)

    # ------------------------------------------------------------------ 준비
    def setup(self, compile_only=False):
        c, S, be = self.c, self.tc.seq_len, self.be
        t0 = time.time()
        self.embed = self.W.embed()                                   # host 에 둔다(gather 만 한다)
        if c.gemma:                                                   # Gemma: sqrt(H)-scaled input embeddings
            if c.window and S > c.window:
                raise ValueError(f"seq_len {S} exceeds the sliding window {c.window}")
            self.embed = self.embed * torch.tensor(c.H ** 0.5, dtype=self.embed.dtype)
        self.dev_norm = self.be_head.to_device(self.W.norm(c))
        self._setup_vocab_chunks()
        cos, sin = L.rope_tables(c, S)
        self.dev_cos, self.dev_sin = be.to_device(cos), be.to_device(sin)
        self._tabs = {}                                   # per layer type (Gemma: sliding layers' own RoPE base)
        for kind in set(c.layer_types):
            lc, ls = L.rope_tables(c, S, kind=kind)
            self._tabs[kind] = (be.to_device(lc), be.to_device(ls))
        self.dev_rot = be.to_device(L.rotate_half_matrix(c.hd))
        # the same tables on every device (index = device); device 0 keeps the single-device attributes
        self._tabs_d, self._rot_d, self._cos_d = [self._tabs], [self.dev_rot], [(self.dev_cos, self.dev_sin)]
        for b in self.bes[1:]:
            self._tabs_d.append({k: (b.to_device(x), b.to_device(y)) for k, (x, y) in
                                 ((k, L.rope_tables(c, S, kind=k)) for k in set(c.layer_types))})
            self._rot_d.append(b.to_device(L.rotate_half_matrix(c.hd)))
            self._cos_d.append((b.to_device(cos), b.to_device(sin)))
        if c.moe:
            from .moe import HostMLP
            tc = self.tc
            self.host_mlp = HostMLP(self.path, trainable=tc.full, lr=tc.lr, betas=tc.betas,
                                    weight_decay=tc.weight_decay)
            self.log(json.dumps({"event": "host_mlp", "layers": len(self.host_mlp.blocks),
                                 "seconds": round(time.time() - t0, 1)}))
            if tc.moe_hot_fraction > 0:
                from . import moe_device as MD
                cp = lambda k, gm, smp: G.compile_parts(self.be, k, gm, smp, self.max_mm, log=lambda m: self.log(
                    json.dumps({"event": "compile_part", "msg": m})))
                self.moe_dev = MD.DeviceExperts(self.be, cp, tc.full)
                self.moe_wraps = MD.attach(self.host_mlp.blocks, self.moe_dev, tc.moe_hot_fraction)
        self.dev_W = []
        for i in range(c.n_layers):
            w = self.W.layer(i, c)
            self.dev_W.append([self.bes[self._dev[i]].to_device(w[k]) for k in L.layer_weights(c)])
        self.log(json.dumps({"event": "weights_on_device", "layers": c.n_layers,
                             "seconds": round(time.time() - t0, 1)}))
        self._init_adapters()
        self._compile()

    def _setup_vocab_chunks(self):
        """lm_head 를 어휘 조각으로 나눠 장치에 올린다(조각 크기·가짜 토큰 bias·조각별 id)."""
        c, be = self.c, self.be_head
        n, vc, pad = L.vocab_chunks(c.vocab)
        lm = self.W.lm_head()
        if pad:
            lm = torch.cat([lm, torch.zeros(pad, c.H, dtype=lm.dtype)])
        ids = torch.arange(n * vc, dtype=torch.float32)
        self.vc = vc
        self.dev_lm = [be.to_device(lm[j * vc:(j + 1) * vc].contiguous()) for j in range(n)]
        self.dev_iota = [be.to_device(ids[j * vc:(j + 1) * vc].view(1, -1).contiguous()) for j in range(n)]
        self.dev_bias = [be.to_device(torch.where(ids[j * vc:(j + 1) * vc] < c.vocab, 0.0, -1e9).view(1, -1))
                         for j in range(n)]
        self.dev_gx0 = be.to_device(torch.zeros(self.tc.seq_len, c.H))

    def _init_adapters(self):
        """학습할 파라미터의 fp32 master 와 옵티마이저를 만든다.

        LoRA: PEFT 기본값과 같게 A 는 kaiming-uniform(a=√5), B 는 0 → 시작 시 모델 출력이 그대로다.
        전체 파인튜닝: 레이어 가중치 9종(LAYER_WEIGHTS)의 fp32 사본. 임베딩·최종 norm·lm_head 는 고정.
        """
        c, r = self.c, self.lora.r
        if self.tc.full:
            self.master = [[self.W.get(f"model.layers.{i}.{L.hf_name(k)}", torch.float32)
                            for k in L.layer_weights(c)] for i in range(c.n_layers)]
        else:
            # 전역 난수에 기대지 않는다: furiosa 를 불러오면 torch.manual_seed 가 장치 쪽에서 경고만 내고
            # 끝나, 같은 seed 로 만든 두 trainer 의 A 가 달라졌다(check 가 엉뚱한 차이를 보고).
            gen = torch.Generator().manual_seed(self.tc.seed)
            self.master = []                 # [layer] = [A_t, B_t, ...] (fp32, targets 순서)
            for _ in range(c.n_layers):
                per = []
                for t in self.lora.targets:
                    d_in, d_out = L.LORA_TARGETS[t](c)
                    bound = 1.0 / math.sqrt(d_in)     # kaiming_uniform(a=√5) 의 경계 = 1/√fan_in (PEFT 기본)
                    per += [(torch.rand(r, d_in, generator=gen) * 2 - 1) * bound, torch.zeros(d_out, r)]
                self.master.append(per)
        self.params = [p for per in self.master for p in per]
        for p in self.params:
            p.requires_grad_(True)
        def mk(ps):
            # CPU fused AdamW(torch 2.4+)는 벡터화·병렬 커널 하나로 돈다 — foreach 보다 여러 배 빠르다.
            # 전체 파인튜닝에서 host 갱신이 다음 순전파를 붙잡는 병목이라 중요하다.
            kw = dict(lr=self.tc.lr, betas=self.tc.betas, weight_decay=self.tc.weight_decay, eps=1e-8)
            try:
                return torch.optim.AdamW(ps, fused=True, **kw)
            except (RuntimeError, TypeError, ValueError):
                return torch.optim.AdamW(ps, foreach=True, **kw)
        if self.tc.full:
            # 레이어마다 따로: 레이어 i 갱신이 끝나는 대로 올려 다음 순전파와 겹친다.
            self.opts = [mk(per) for per in self.master]
            self._ready = [None] * c.n_layers           # 레이어별 업로드 future (다음 순전파가 기다린다)
            # 레이어마다 [내리기 → (clip 확정 대기) → AdamW → 올리기] 를 한 작업으로, 여러 레이어를 병렬로.
            self._layer_pool = ThreadPoolExecutor(4, thread_name_prefix="npu-lora-layer")
            self._step_ev = None
            self._bufs = [None] * c.n_layers            # 레이어별 host 버퍼 (bf16 기울기, fp32 기울기, bf16 업로드)
            self._rstate = {}                           # resident layer -> [(w32, m, v) on the device | None (1-D)]
            for i in sorted(self._resident):
                b = self.bes[self._dev[i]]
                self._rstate[i] = [(b.to_device(p.detach().contiguous()), b.to_device(torch.zeros_like(p)),
                                    b.to_device(torch.zeros_like(p))) if p.dim() >= 2 else None
                                   for p in self.master[i]]
        else:
            self.opt = mk(self.params)
        self._upload_adapters()

    def _upload_adapters(self):
        """갱신된 파라미터를 bf16 으로 장치에 올린다(LoRA: 어댑터, 전체: 레이어 가중치)."""
        with torch.no_grad():
            if self.tc.full:
                self.dev_W = [[self.bes[self._dev[i]].to_device(p.detach().to(BF)) for p in per]
                              for i, per in enumerate(self.master)]
                self.dev_AD = [[] for _ in self.master]
            else:
                self.dev_AD = [[self.be.to_device(p.detach().to(BF)) for p in per] for per in self.master]

    def _samples(self):
        c, S, r = self.c, self.tc.seq_len, self.lora.r
        h = torch.zeros(S, c.H, dtype=BF)
        w = [torch.zeros_like(x) for x in self.W.layer(0, c).values()]
        ad = []
        for t in (() if self.tc.full else self.lora.targets):
            d_in, d_out = L.LORA_TARGETS[t](c)
            ad += [torch.zeros(r, d_in, dtype=BF) + 0.01, torch.zeros(d_out, r, dtype=BF) + 0.01]
        cos, sin = L.rope_tables(c, S)
        mask = L.causal_mask(S)
        return h, w, ad, cos, sin, mask, L.rotate_half_matrix(c.hd)

    def _compile(self):
        c, S, lc = self.c, self.tc.seq_len, self.lora
        h, w, ad, cos, sin, mask, rot = self._samples()
        key = (f"llama-v2-H{c.H}-I{c.I}-nh{c.nh}-kv{c.nkv}{c.key}-S{S}-full" if self.tc.full else
               f"llama-v2-H{c.H}-I{c.I}-nh{c.nh}-kv{c.nkv}{c.key}-S{S}-r{lc.r}-{''.join(lc.targets)}-s{lc.scale}")
        targets, scale = ((), 1.0) if self.tc.full else (lc.targets, lc.scale)
        lay_in = [h, *w, *ad, cos, sin, mask, rot]
        t0 = time.time()
        log = lambda m: self.log(json.dumps({"event": "compile_part", "msg": m}))
        cpd = lambda b: (lambda k, gm, smp: G.compile_parts(b, k, gm, smp, self.max_mm, log=log))
        cp = cpd(self.be_head)
        gm_fwd = G.build(G.layer_fwd_fn(c, targets, scale), lay_in)
        bwd_in = lay_in + [h + 0.01]
        bwd_fn = G.layer_bwd_full_fn(c) if self.tc.full else G.layer_bwd_fn(c, targets, scale)
        gm_bwd = G.build(bwd_fn, bwd_in)
        if self.tc.full and self.tc.accum > 1:
            acc_in = bwd_in + [torch.zeros_like(x) + 0.01 for x in w]
            gm_acc = G.build(G.layer_bwd_full_acc_fn(c), acc_in)
        self.g_fwd_d, self.g_bwd_d, self.g_acc_d, self.g_sq_d, self.adamw_d = [], [], [], [], []
        for d, b in enumerate(self.bes):                 # one copy of each layer graph per device
            cpb = cpd(b)
            self.g_fwd_d.append(cpb(f"{key}-fwd", gm_fwd, lay_in))
            self.g_bwd_d.append(cpb(f"{key}-bwd", gm_bwd, bwd_in))
            sq = {}
            if self.tc.full:
                for x in w:
                    shp = tuple(x.shape)
                    if x.dim() < 2 or shp in sq:         # 1차원(norm 가중치)은 작아서 host 에서 바로 잰다
                        continue
                    sq[shp] = cpb(f"sqnorm-{'x'.join(map(str, shp))}", G.build(G.sqnorm_fn(), [x + 0.01]), [x + 0.01])
            self.g_sq_d.append(sq)
            if self.tc.full and self.tc.accum > 1:
                self.g_acc_d.append(cpb(f"{key}-bwd-acc", gm_acc, acc_in))
            if any(self._dev[i] == d for i in self._resident):
                from .optim_device import ResidentAdamW
                ra = ResidentAdamW(b, log=log)
                for x in w:
                    if x.dim() >= 2:
                        ra.compile(tuple(x.shape))
                self.adamw_d.append(ra)
            else:
                self.adamw_d.append(None)
        self.g_fwd, self.g_bwd = self.g_fwd_d[0], self.g_bwd_d[0]
        if self.tc.full:
            self.g_sq = self.g_sq_d[0]
            if self.tc.accum > 1:
                self.g_bwd_acc = self.g_acc_d[0]
        # head: rms → 어휘 조각별 통계/기울기 → rms 역전파 (조각 그래프는 모든 조각이 같은 모양)
        norm, vc = self.W.norm(self.c), self.vc
        x = torch.zeros(S, c.H, dtype=BF) + 0.01
        lm_j = torch.zeros(vc, c.H, dtype=BF) + 0.01
        bias_j, iota_j = torch.zeros(1, vc), torch.arange(vc, dtype=torch.float32).view(1, -1)
        lab, col = torch.zeros(S, 1), torch.ones(S, 1)
        hk = f"llama-v3-H{c.H}-S{S}-Vc{vc}"   # v3: lse·w 중복 입력으로 틀리게 컴파일된 캐시를 버린다
        self.g_rms = cp(f"{hk}-rms", G.build(G.head_rms_fn(c), [h, norm]), [h, norm])
        st_in = [x, lm_j, bias_j, iota_j, lab]
        self.g_stats = cp(f"{hk}-ce-stats", G.build(G.ce_stats_fn(), st_in), st_in)
        gr_in = [x, lm_j, bias_j, iota_j, lab, col, col.clone(), torch.zeros(S, c.H)]
        self.g_cegrad = cp(f"{hk}-ce-grad", G.build(G.ce_grad_fn(), gr_in), gr_in)
        rb_in = [h, norm, torch.zeros(S, c.H)]
        self.g_rms_bwd = cp(f"{hk}-rms-bwd", G.build(G.head_rms_bwd_fn(c), rb_in), rb_in)
        self.log(json.dumps({"event": "compiled", "seconds": round(time.time() - t0, 1), **self.be.describe()}))

    # ------------------------------------------------------------------ 한 step
    def _cs(self, i):
        """(cos, sin) device tables for layer i"""
        kinds, d = self.c.layer_types, self._dev[i]
        return self._tabs_d[d][kinds[i]] if kinds else self._cos_d[d]

    def forward_backward(self, ids, labels, doc_ids=None):
        """ids, labels: [S] (labels 는 다음 토큰 id, 학습 제외 위치는 -1). 반환: (loss, grads)"""
        be, c = self.be, self.c
        n_tok = int((labels >= 0).sum())
        mask = be.to_device(L.causal_mask(len(ids), doc_ids))
        h = be.to_device(self.embed[ids])
        hs, mids = [h], []
        mlp = self.host_mlp
        for i in range(c.n_layers):
            h = self.g_fwd(h, *self.dev_W[i], *self.dev_AD[i], *self._cs(i), mask, self.dev_rot)
            if mlp is not None:                       # MoE: attention on the device, MLP half on the host
                a = be.to_host(h).to(BF)
                mids.append(a)
                h = be.to_device(mlp.forward(i, a))
            hs.append(h)
        loss, g = self._head(hs[-1], labels, n_tok)
        grads = [None] * c.n_layers
        for i in reversed(range(c.n_layers)):
            if mlp is not None:
                g = be.to_device(mlp.backward(i, mids[i], be.to_host(g)).to(BF))
            out = self.g_bwd(hs[i], *self.dev_W[i], *self.dev_AD[i], *self._cs(i), mask,
                             self.dev_rot, g)
            g, grads[i] = out[0], out[1:]
        be.sync()
        grads = [[be.to_host(x).float() for x in gl] for gl in grads]
        return loss, grads, n_tok

    def _head(self, h, labels, n_tok):
        """다음 토큰 cross-entropy 를 어휘 조각으로: (loss, hidden 기울기[장치])."""
        be = self.be_head
        lab = be.to_device(labels.float().view(-1, 1))
        x = self.g_rms(h, self.dev_norm)
        stats = []
        for lm_j, bias_j, iota_j in zip(self.dev_lm, self.dev_bias, self.dev_iota):
            m, s, t = self.g_stats(x, lm_j, bias_j, iota_j, lab)
            stats.append((be.to_host(m).float(), be.to_host(s).float(), be.to_host(t).float()))
        lse, tgt = L.ce_combine(stats)
        valid = (labels >= 0).float().view(-1, 1)
        w = valid / max(n_tok, 1)
        loss = float(((lse - tgt) * w).sum())
        lse_d, w_d = be.to_device(lse), be.to_device(w)
        gx = self.dev_gx0
        for lm_j, bias_j, iota_j in zip(self.dev_lm, self.dev_bias, self.dev_iota):
            gx = self.g_cegrad(x, lm_j, bias_j, iota_j, lab, lse_d, w_d, gx)
        return loss, self.g_rms_bwd(h, self.dev_norm, gx)

    def train_step(self, batch):
        """batch: [(ids, labels, doc_ids)] × accum. 반환: 지표 dict"""
        if self.tc.full:
            return self._train_step_full(batch)
        t0 = time.time()
        acc = None
        losses, toks = [], 0
        for ids, labels, doc_ids in batch:
            loss, grads, n = self.forward_backward(ids, labels, doc_ids)
            losses.append(loss)
            toks += len(ids)
            acc = grads if acc is None else [[a + b for a, b in zip(x, y)] for x, y in zip(acc, grads)]
        k = len(batch)
        acc = [[x / k for x in gl] for gl in acc]
        gnorm = math.sqrt(sum(float(x.pow(2).sum()) for gl in acc for x in gl))
        clip = min(1.0, self.tc.grad_clip / (gnorm + 1e-6)) if self.tc.grad_clip else 1.0
        self.step_no += 1
        lr = self._lr()
        with torch.no_grad():
            for p, g in zip(self.params, (x for gl in acc for x in gl)):
                p.grad = g.to(p.dtype) * clip
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self._upload_adapters()
        self._moe_replan()
        dt = time.time() - t0
        return {"step": self.step_no, "loss": sum(losses) / k, "grad_norm": gnorm, "lr": lr,
                "tokens": toks, "seconds": round(dt, 3), "tokens_per_s": round(toks / dt, 1)}

    # ------------------------------------------------------------------ 전체 파인튜닝 (offload 파이프라인)
    def sync_params(self):
        """밀린 갱신을 모두 반영한다(저장·평가 전에)."""
        if self.tc.full:
            for i in range(self.c.n_layers):
                self._layer_w(i)
            self._pull_resident()

    def _layer_job(self, i, dev_grads, k, ev, box):
        """레이어 i: 기울기 내리기 → clip 확정 대기 → host AdamW → bf16 올리기. 새 장치 가중치를 돌려준다."""
        t0 = time.time()
        be = self.bes[self._dev[i]]
        if self._bufs[i] is None:
            self._bufs[i] = ([be.host_buffer(x) for x in dev_grads],
                             [torch.empty(p.shape, dtype=torch.float32) for p in self.master[i]],
                             [be.host_buffer(torch.empty(p.shape, dtype=BF)) for p in self.master[i]])
        hb, g32, up = self._bufs[i]
        grads = [g.copy_(be.to_host(x, out=b)).div_(k) for x, b, g in zip(dev_grads, hb, g32)]
        t1 = time.time()
        # this step's backward no longer needs layer i: drop the old bf16 device weights before uploading the
        # new ones, or every layer briefly holds two copies (8B: +14 GB, out of device memory after ~15 steps)
        self.dev_W[i] = None
        ev.wait()                                   # 역전파가 끝나야 전역 clip 계수가 정해진다
        clip, lr = box
        with torch.no_grad():
            for p, g in zip(self.master[i], grads):
                p.grad = g.mul_(clip) if clip != 1.0 else g
        for group in self.opts[i].param_groups:
            group["lr"] = lr
        self.opts[i].step()
        self.opts[i].zero_grad(set_to_none=True)
        t2 = time.time()
        with torch.no_grad():
            out = [be.to_device(u.copy_(p)) for p, u in zip(self.master[i], up)]
        t3 = time.time()
        return out, (t1 - t0, t2 - t1, t3 - t2)

    def _resident_update(self, i, dev_grads, k, clip, lr):
        """Resident layer i: AdamW on the device for 2-D weights (state never leaves the card); the small 1-D
        norm weights go through the host optimizer. Returns the new bf16 device weights."""
        from .optim_device import scalars
        be, ra = self.bes[self._dev[i]], self.adamw_d[self._dev[i]]
        t0 = time.time()
        tc = self.tc
        sc = [be.to_device(x) for x in scalars(lr, tc.betas, 1e-8, tc.weight_decay, self.step_no)]
        gs = be.to_device(torch.tensor([[clip / k]], dtype=torch.float32))
        out, st = [], self._rstate[i]
        with torch.no_grad():
            for j, (x, s) in enumerate(zip(dev_grads, st)):
                if s is None:
                    p = self.master[i][j]
                    p.grad = be.to_host(x).float().mul_(clip / k)
                    out.append(None)
                    continue
                w2, m2, v2, wbf = ra.step(s, x, gs, sc)
                st[j] = (w2, m2, v2)
                out.append(wbf)
        for group in self.opts[i].param_groups:
            group["lr"] = lr
        self.opts[i].step()                         # only the 1-D params have .grad
        self.opts[i].zero_grad(set_to_none=True)
        with torch.no_grad():
            out = [be.to_device(self.master[i][j].detach().to(BF)) if o is None else o for j, o in enumerate(out)]
        return out, (0.0, time.time() - t0, 0.0)

    def _pull_resident(self):
        """Copy the device-resident fp32 masters back into self.master (before export)."""
        for i, st in getattr(self, "_rstate", {}).items():
            be = self.bes[self._dev[i]]
            with torch.no_grad():
                for p, s in zip(self.master[i], st):
                    if s is not None:
                        p.copy_(be.to_host(s[0]).float())

    def _update_layer(self, i, grads, clip, lr):
        """(테스트·기준용) host AdamW 로 레이어 i 를 갱신하고 bf16 장치 텐서를 돌려준다."""
        with torch.no_grad():
            for p, g in zip(self.master[i], grads):
                p.grad = g.mul_(clip) if clip != 1.0 else g
        for group in self.opts[i].param_groups:
            group["lr"] = lr
        self.opts[i].step()
        self.opts[i].zero_grad(set_to_none=True)
        with torch.no_grad():
            return [self.be.to_device(p.detach().to(BF)) for p in self.master[i]]

    def _layer_w(self, i):
        """레이어 i 의 장치 가중치. 직전 step 의 갱신·업로드가 아직이면 끝날 때까지 기다린다."""
        f = self._ready[i]
        if f is not None:
            self.dev_W[i], self._last_times[i] = f.result()
            self._ready[i] = None
        return self.dev_W[i]

    def _train_step_full(self, batch):
        import threading
        be, c = self.be, self.c
        t0 = time.time()
        k = len(batch)
        acc = [None] * c.n_layers
        sq = [None] * c.n_layers
        ev, box = threading.Event(), [1.0, 0.0]
        bes, dev = self.bes, self._dev
        jobs = [None] * c.n_layers
        losses, toks, t_wait = [], 0, 0.0
        mlp = self.host_mlp                             # MoE: norm + MoE block on the host, trained there
        if not hasattr(self, "_last_times"):
            self._last_times = [(0.0, 0.0, 0.0)] * c.n_layers
        for mi, (ids, labels, doc_ids) in enumerate(batch):
            last = mi == k - 1
            n_tok = int((labels >= 0).sum())
            cm = L.causal_mask(len(ids), doc_ids)
            masks = [b.to_device(cm) for b in bes]
            h = bes[dev[0]].to_device(self.embed[ids])
            hs, mids = [h], []
            for i in range(c.n_layers):
                d = dev[i]
                if i and d != dev[i - 1]:              # hidden state to the next device (S x H bf16)
                    h = bes[d].to_device(bes[dev[i - 1]].to_host(h))
                tw = time.time()
                w = self._layer_w(i)
                t_wait += time.time() - tw
                h = self.g_fwd_d[d](h, *w, *self._cs(i), masks[d], self._rot_d[d])
                if mlp is not None:
                    a = be.to_host(h).to(BF)
                    mids.append(a)
                    h = be.to_device(mlp.forward(i, a))
                hs.append(h)
            loss, g = self._head(hs[-1], labels, n_tok)
            for i in reversed(range(c.n_layers)):
                d = dev[i]
                if i < c.n_layers - 1 and d != dev[i + 1]:  # gradient back to the previous device
                    g = bes[d].to_device(bes[dev[i + 1]].to_host(g))
                if mlp is not None:
                    g = be.to_device(mlp.backward(i, mids[i], be.to_host(g)).to(BF))
                tabs = (*self._cs(i), masks[d], self._rot_d[d])
                if acc[i] is None:
                    out = self.g_bwd_d[d](hs[i], *self.dev_W[i], *tabs, g)
                else:
                    out = self.g_acc_d[d](hs[i], *self.dev_W[i], *tabs, g, *acc[i])
                g, acc[i] = out[0], list(out[1:])
                if last:
                    # 크기는 장치에서(스칼라 9개), 큰 기울기는 다른 스레드에서 내리기 시작
                    sq[i] = [self.g_sq_d[d][tuple(x.shape)](x) if x.dim() >= 2 else x for x in acc[i]]
                    if i not in self._resident:
                        jobs[i] = self._layer_pool.submit(self._layer_job, i, acc[i], k, ev, box)
            losses.append(loss)
            toks += len(ids)
        t_bwd = time.time()
        total = sum(float(bes[dev[i]].to_host(x).float().sum()) if x.dim() >= 2 and x.shape[-1] == 1
                    else float(bes[dev[i]].to_host(x).float().pow(2).sum())
                    for i, per in enumerate(sq) for x in per) / (k * k)
        if mlp is not None:
            total += mlp.sq_norm() / (k * k)
        gnorm = math.sqrt(total)
        clip = min(1.0, self.tc.grad_clip / (gnorm + 1e-6)) if self.tc.grad_clip else 1.0
        self.step_no += 1
        lr = self._lr()
        box[0], box[1] = clip, lr
        if mlp is not None:
            mlp.step(clip, lr, k)
            if self.moe_dev is not None:
                for w in self.moe_wraps:
                    if w is not None:
                        self.moe_dev.refresh(w.layer, w.inner)
        self._moe_replan()
        ev.set()                                    # 기다리던 레이어 작업들이 갱신·올리기를 시작한다
        t_res = time.time()
        for i in sorted(self._resident):            # device AdamW: graphs run in this thread, card by card
            self.dev_W[i], self._last_times[i] = self._resident_update(i, acc[i], k, clip, lr)
        t_res = time.time() - t_res
        for i in range(c.n_layers):
            self._ready[i] = jobs[i]                # 다음 step 의 순전파가 레이어별로 기다린다
        dt = time.time() - t0
        prev = [sum(x[j] for x in self._last_times) for j in range(3)]
        return {"step": self.step_no, "loss": sum(losses) / k, "grad_norm": gnorm, "lr": lr,
                "tokens": toks, "seconds": round(dt, 3), "tokens_per_s": round(toks / dt, 1),
                "wait_weights_s": round(t_wait, 3), "norm_s": round(time.time() - t_bwd, 3),
                "prev_d2h_s": round(prev[0], 3), "prev_adamw_s": round(prev[1], 3), "prev_upload_s": round(prev[2], 3),
                **({"resident_update_s": round(t_res, 3)} if self._resident else {}),
                **({"devices": len(bes)} if len(bes) > 1 else {})}

    def _moe_replan(self):
        if self.moe_dev is not None and self.step_no % max(self.tc.moe_replan_every, 1) == 0:
            from . import moe_device as MD
            MD.replan(self.moe_wraps, self.moe_dev)

    def moe_stats(self):
        """hot experts per layer and device expert calls so far (MoE with moe_hot_fraction)"""
        if self.moe_dev is None:
            return {}
        return {"hot_per_layer": [len(w.hot) if w else 0 for w in self.moe_wraps], "device_calls": self.moe_dev.calls}

    def _lr(self):
        tc = self.tc
        if self.step_no <= tc.warmup:
            return tc.lr * self.step_no / max(tc.warmup, 1)
        prog = (self.step_no - tc.warmup) / max(tc.steps - tc.warmup, 1)
        return tc.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    def adapters(self):
        """{(layer, target): (A fp32, B fp32)} — LoRA 모드에서만."""
        assert not self.tc.full, "full fine-tuning has no adapters; use export.save_full"
        return {(i, t): (per[2 * j].detach(), per[2 * j + 1].detach())
                for i, per in enumerate(self.master) for j, t in enumerate(self.lora.targets)}

    def layer_weights(self):
        """{(layer, LAYER_WEIGHTS 키): fp32 가중치} — 전체 파인튜닝 모드에서만."""
        assert self.tc.full
        self.sync_params()
        return {(i, k): p.detach() for i, per in enumerate(self.master) for k, p in zip(L.layer_weights(self.c), per)}

    def summary(self):
        return {"model": self.path, "config": asdict(self.c), "lora": {**asdict(self.lora), "scale": self.lora.scale},
                "train": asdict(self.tc), "graph_parts_max_mm": self.max_mm, **self.be.describe()}

