"""학습 그래프 만들기: 순전파+역전파를 aten 연산만 남은 FX 그래프로 펼친다.

NPU 컴파일러는 "그래프 하나"만 받는다. torch.func.vjp 를 make_fx 로 추적하면 역전파 연산이
평범한 순전파 연산처럼 들어간 그래프가 나오고, 그걸 그대로 컴파일하면 된다.
그 과정에서 컴파일러가 받지 못하는 패턴은 의미를 바꾸지 않는 변환으로 걸러 낸다.
"""
import hashlib
import time

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from . import llama as L

POINTWISE = {"aten.mul.Tensor", "aten.div.Tensor", "aten.add.Tensor", "aten.sub.Tensor",
             "aten.mul.Scalar", "aten.div.Scalar", "aten.add.Scalar", "aten.sub.Scalar",
             "aten.neg.default", "aten._to_copy.default", "aten.where.self"}


def drop_broadcast_expands(gm):
    """원소별 연산에만 쓰이는 expand 를 지우고 브로드캐스트에 맡긴다.

    reduction 의 역전파는 [1,1] → [S,H] expand(stride 0)를 만든다. 그 텐서가 원소별 연산을 두 번
    거쳐 일반 텐서와 곱해지면 furiosa global-compiler 가 패닉한다(itertools zip_eq). 소비자가 모두
    원소별 연산이면 지워도 결과 모양은 브로드캐스트로 같고, build() 가 CPU 재실행으로 확인한다.
    """
    n = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and str(node.target) == "aten.expand.default":
            if all(u.op == "call_function" and str(u.target) in POINTWISE for u in node.users):
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                n += 1
    gm.graph.lint()
    gm.recompile()
    return n


FRESH = {"aten.empty_like.default", "aten.zeros_like.default", "aten.empty.memory_format",
         "aten.new_empty.default"}


def _shape(n):
    v = n.meta.get("val") if hasattr(n, "meta") else None
    return tuple(v.shape) if v is not None else None


def scalarize_constants(gm):
    """상수 텐서 c = fill(empty_like(x), v) 를 쓰는 원소별 연산을 스칼라 연산으로 바꾼다.

    SiLU·sigmoid 역전파는 1 - sigmoid(x) 의 1 을 empty_like + fill 로 만든다. furiosa 컴파일러는
    이 상수 텐서가 든 그래프에서 UnsupportedOpError 를 낸다. 상수와 같은 모양의 텐서 y 와 만나는
    자리만 바꾸므로 결과 모양은 그대로다:
        c + y → y + v      c - y → -y + v      y - c → y + (-v)
        c * y → y * v      y / c → y * (1/v)
    바꿀 수 없는 소비자가 남으면 ones_like(x) * v 로 둔다.
    """
    aten = torch.ops.aten
    n = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) not in ("aten.fill.Scalar", "aten.fill_.Scalar"):
            continue
        src, v = node.args[0], node.args[1]
        if not (src.op == "call_function" and str(src.target) in FRESH):
            continue
        cshape = _shape(node)
        for user in list(node.users):
            t = str(user.target)
            if user.op != "call_function" or len(user.args) != 2:
                continue
            a, b = user.args
            other = b if a is node else a
            if other is node or _shape(other) != cshape or _shape(user) != cshape:
                continue
            with gm.graph.inserting_before(user):
                g = gm.graph
                if t == "aten.add.Tensor":
                    new = g.call_function(aten.add.Scalar, (other, v))
                elif t == "aten.mul.Tensor":
                    new = g.call_function(aten.mul.Scalar, (other, v))
                elif t == "aten.sub.Tensor" and a is node:
                    new = g.call_function(aten.add.Scalar, (g.call_function(aten.neg.default, (other,)), v))
                elif t == "aten.sub.Tensor":
                    new = g.call_function(aten.add.Scalar, (other, -v))
                elif t == "aten.div.Tensor" and b is node and v != 0:
                    new = g.call_function(aten.mul.Scalar, (other, 1.0 / v))
                else:
                    continue
            new.meta["val"] = user.meta.get("val")
            user.replace_all_uses_with(new)
            g.erase_node(user)
            n += 1
        if node.users:                      # 남은 소비자가 있으면 ones_like * v 로
            with gm.graph.inserting_before(node):
                ones = gm.graph.call_function(aten.ones_like.default, (src.args[0],))
                new = gm.graph.call_function(aten.mul.Scalar, (ones, v))
            node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        if not src.users:
            gm.graph.erase_node(src)
    gm.graph.lint()
    gm.recompile()
    return n


def strip_device_kwargs(gm):
    """ones_like(…, device=cpu) 처럼 추적 때 박힌 device 인자를 지운다 — 입력 텐서의 장치를 따르게.

    그래프는 CPU 샘플로 추적하므로 vjp 의 시작 기울기(ones_like(loss))가 device=cpu 를 기록한다.
    CUDA 에서 그대로 돌리면 역전파가 CPU 텐서에서 시작해 가중치(GPU)와 곱하다 실패한다.
    """
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and "device" in node.kwargs:
            kw = dict(node.kwargs)
            kw.pop("device")
            kw.pop("pin_memory", None)
            node.kwargs = kw
            n += 1
    gm.recompile()
    return n


def decompose_softmax_backward(gm):
    """_softmax_backward_data(g, y, dim) → y * (g - mean(g*y, dim) * N).

    softmax 출력 y 가 그래프(조각) 입력으로 들어오면 furiosa 가 이 연산을 mul+reduce 융합 커널로
    만들다 "verification of operator failed (EinsumByDpe: input_descriptors 1 != 2)" 로 멈춘다.
    합을 mean × N 으로 쓰면 같은 값이고 컴파일된다(probe6: 상대오차 3e-4, bf16 기울기 수준).
    """
    aten = torch.ops.aten
    n = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != "aten._softmax_backward_data.default":
            continue
        g, y, dim = node.args[0], node.args[1], node.args[2]
        size = y.meta["val"].shape[dim]
        with gm.graph.inserting_before(node):
            gr = gm.graph
            gy = gr.call_function(aten.mul.Tensor, (g, y))
            mean = gr.call_function(aten.mean.dim, (gy, [dim], True))
            s = gr.call_function(aten.mul.Scalar, (mean, float(size)))
            d = gr.call_function(aten.sub.Tensor, (g, s))
            new = gr.call_function(aten.mul.Tensor, (y, d))
        new.meta["val"] = node.meta.get("val")
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        n += 1
    gm.graph.lint()
    gm.recompile()
    return n


PASSES = (drop_broadcast_expands, scalarize_constants, strip_device_kwargs, decompose_softmax_backward)


def build(fn, sample, check=True):
    """fn(*sample) 을 추적하고 PASSES 를 적용한 GraphModule. check 면 변환 전후 출력을 비교한다.

    functionalize 로 in-place 연산(fill_, sub_ …)을 먼저 없앤다 — 컴파일러는 순전파 그래프만
    받아 왔고, 역전파 그래프에서 처음 보는 in-place 연산은 지원 밖일 가능성이 크다.
    """
    # 같은 텐서 객체가 두 자리에 오면 make_fx 가 두 입력을 하나로 묶어 버린다(그래프가 한쪽만 쓴다).
    # 샘플 재사용은 흔한 실수라 여기서 복제해 막는다.
    _, sample = dedup_tensors(sample)
    gm = make_fx(torch.func.functionalize(fn, remove="mutations"))(*sample)
    before = gm(*sample) if check else None
    applied = {p.__name__: p(gm) for p in PASSES}
    if check:
        after = gm(*sample)
        for i, (x, y) in enumerate(zip(_tup(before), _tup(after))):
            x, y = x.float(), y.float()
            bad = x.shape != y.shape or bool(torch.isnan(y).any() & ~torch.isnan(x).any())
            rel = float((x - y).norm() / (x.norm() + 1e-12)) if not bad else float("inf")
            if bad or rel > 1e-2:
                raise RuntimeError(f"graph pass changed output {i}: rel={rel:.3g} ({applied})")
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        if node.op == "output":
            continue
        if isinstance(val, torch.Tensor) and val.dim() == 0 and node.op == "call_function":
            # 0차원 중간값은 컴파일러 패닉의 다른 원인이다 — 남아 있으면 미리 알린다.
            print(f"[graph] warning: 0-d intermediate {node.name} ({node.target})")
    return gm


def _tup(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def dedup_tensors(args):
    """같은 텐서 객체가 두 자리에 있으면 뒤쪽을 복제한다. (복제한 개수, 새 인자) 를 돌려준다.

    make_fx 도, CompileModule.from_module(torch.export) 도 같은 객체를 받은 두 입력을 하나로 묶는다.
    컴파일된 모듈은 실행 때 한쪽 값을 두 자리에 쓴다 — 모양이 같으면 오류 없이 틀린다
    (어휘 조각 기울기: lse 와 w 가 묶여 exp(logits - w) 로 폭증).
    """
    seen, out, n = set(), [], 0
    for a in args:
        if isinstance(a, torch.Tensor) and id(a) in seen:
            a = a.clone()
            n += 1
        seen.add(id(a))
        out.append(a)
    return n, out


# ---------------------------------------------------------------- 그래프 조각내기

MATMUL = {"aten.mm.default", "aten.bmm.default", "aten.matmul.default", "aten.addmm.default"}
VIEWS = {"aten.t.default", "aten.view.default", "aten._unsafe_view.default", "aten.transpose.int",
         "aten.unsqueeze.default", "aten.squeeze.dim", "aten.expand.default", "aten.permute.default",
         "aten.slice.Tensor", "_operator.getitem", "aten.alias.default"}


def plan_parts(gm, max_mm):
    """call_function 노드 → 조각 번호. max_mm 번째 행렬곱 바로 뒤에서 자른다.

    furiosa 컴파일러는 큰 융합 그래프에서 시간이 폭증하거나("attention 순전파 15분 초과")
    내부 오류를 낸다("memory explosion": RMSNorm 역전파 + SwiGLU 역전파를 합친 그래프). 각각은
    따로 컴파일된다. 행렬곱 뒤에서 자르면 조각마다 계산이 들어가고, view 만 남은 조각이 생기지 않는다.
    """
    part, mm, assign, last_compute = 0, 0, {}, {}
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        assign[n] = part
        t = str(n.target)
        if t not in VIEWS:
            last_compute[part] = True
        if t in MATMUL:
            mm += 1
            if mm >= max_mm:
                part, mm = part + 1, 0
    # view 만 들어간 조각(주로 끝 조각)은 앞 조각에 붙인다
    for n, p in list(assign.items()):
        if not last_compute.get(p) and p > 0:
            assign[n] = p - 1
    return split_disconnected(assign)


def split_disconnected(assign):
    """한 조각 안에서 서로 이어지지 않은 계산 덩어리를 각자 다른 조각으로 나눈다.

    furiosa 는 연결되지 않은 부분 그래프가 둘 이상인 그래프를 받지 않는다("multiple internal
    subgraphs are not supported"). head 의 one-hot(iota == labels)처럼 나중에야 합쳐지는 계산이
    같은 조각에 들어가면 생긴다. 같은 조각 안의 덩어리끼리는 서로 의존하지 않으므로 어떤
    순서로 나눠도 조각 사이 의존은 그대로 비순환이다.
    """
    by_part = {}
    for n, p in assign.items():
        by_part.setdefault(p, []).append(n)
    out, next_id = {}, 0
    for p in sorted(by_part):
        nodes = by_part[p]
        members = set(nodes)
        parent = {n: n for n in nodes}

        def find(x):
            while parent[x] is not x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for n in nodes:
            for a in n.all_input_nodes:
                if a in members:
                    parent[find(a)] = find(n)
        comp_id = {}
        for n in nodes:                       # 덩어리 번호는 처음 나온 순서대로
            r = find(n)
            if r not in comp_id:
                comp_id[r] = next_id
                next_id += 1
            out[n] = comp_id[r]
    return out


def materialize_outputs(gm, outs):
    """연속(contiguous) 배치가 아닌 출력 앞에 contiguous 복사를 넣는다. 고친 출력 수를 돌려준다.

    furiosa 컴파일 결과는 출력 텐서를 연속 배치로 가정하고 돌려준다. 그래서 출력이 전치 view 이거나
    전치된 텐서에 원소별 연산을 한 결과면, 직사각이면 모양이 뒤바뀌고 정사각이면 **조용히** 전치된
    값이 나온다(rel err 1.41 = √2). clone(memory_format=contiguous) 은 실제 재배치로 컴파일된다.
    """
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    args = out_node.args[0]
    single = not isinstance(args, (tuple, list))
    args = [args] if single else list(args)
    fixed = 0
    for i, (a, t) in enumerate(zip(args, outs)):
        if isinstance(t, torch.Tensor) and not t.is_contiguous():
            with gm.graph.inserting_before(out_node):
                args[i] = gm.graph.call_function(torch.ops.aten.clone.default, (a,),
                                                 {"memory_format": torch.contiguous_format})
            fixed += 1
    if fixed:
        out_node.args = (args[0] if single else type(out_node.args[0])(args),)
        gm.recompile()
    return fixed


class _Compiled(torch.nn.Module):
    """컴파일된 조각을 원래 조각과 같은 출력 구조(텐서 / 튜플)로 돌려준다."""

    def __init__(self, fn, as_tuple):
        super().__init__()
        self.fn, self.as_tuple = fn, as_tuple

    def forward(self, *args):
        out = self.fn(*args)
        seqout = isinstance(out, (tuple, list))
        if self.as_tuple:
            return tuple(out) if seqout else (out,)
        return out[0] if seqout else out


def compile_parts(backend, key, gm, sample, max_mm=2, log=print):
    """gm 을 조각내 조각마다 backend 로 컴파일한 실행 가능한 모듈. max_mm=0 이면 통째로."""
    if not max_mm:
        _, sample = dedup_tensors(a.contiguous() if isinstance(a, torch.Tensor) else a for a in sample)
        with torch.no_grad():
            materialize_outputs(gm, gm(*sample))
        return backend.compile(key, gm, sample)
    from torch.fx.passes.split_module import split_module
    assign = plan_parts(gm, max_mm)
    split = split_module(gm, None, lambda n: assign[n], keep_original_order=True)
    names = [n for n, _ in split.named_children() if n.startswith("submod_")]
    captured = {}
    hooks = [getattr(split, n).register_forward_pre_hook(
        lambda m, a, n=n: captured.__setitem__(n, a)) for n in names]
    with torch.no_grad():
        split(*sample)                          # 조각마다 입력 모양을 얻는다(CPU)
    for h in hooks:
        h.remove()
    for i, n in enumerate(names):
        sub = getattr(split, n)
        # 컴파일된 커널은 샘플 입력의 메모리 배치를 가정하고 굳는다. 실행 때 조각 입력은 항상 연속
        # 배치(앞 조각 출력은 materialize_outputs 로, 바깥 입력은 host 연속 텐서)이므로 샘플도 맞춘다.
        # 전치된 샘플로 컴파일하면 연속 입력을 받았을 때 메모리를 엉뚱하게 읽어 조용히 틀린다
        # (q·k 조각: 입력 오차 0.4% → 출력 123%).
        n_dup, args = dedup_tensors(a.contiguous() if isinstance(a, torch.Tensor) else a for a in captured[n])
        args = tuple(args)
        with torch.no_grad():
            out = sub(*args)
        materialize_outputs(sub, out)
        sig = sub.code + repr([(tuple(a.shape), str(a.dtype), a.stride()) for a in args if isinstance(a, torch.Tensor)])
        if n_dup:                       # 중복 입력으로 컴파일됐던 예전 캐시(틀린 결과)를 쓰지 않게
            sig += f"|dedup{n_dup}"
        code_hash = hashlib.blake2b(sig.encode(), digest_size=6).hexdigest()
        t0 = time.time()
        fn = backend.compile(f"{key}-p{i}of{len(names)}-{code_hash}", sub, args)
        setattr(split, n, _Compiled(fn, isinstance(out, (tuple, list))))
        log(f"[parts] {key} {i + 1}/{len(names)} "
            f"({sum(1 for x in sub.graph.nodes if x.op == 'call_function')} ops) {time.time() - t0:.0f}s")
    return split


class GraphModule(torch.nn.Module):
    """CompileModule.from_module 이 받을 수 있게 FX 그래프를 nn.Module 로 감싼다."""

    def __init__(self, gm):
        super().__init__()
        self.gm = gm

    def forward(self, *args):
        return self.gm(*args)


# ---------------------------------------------------------------- 레이어/헤드 학습 그래프

def _split(c, targets, args):
    """평평한 인자 → (h, weights dict, lora dict, (cos, sin, mask, rot), rest)"""
    h = args[0]
    nw = len(L.LAYER_WEIGHTS)
    w = dict(zip(L.LAYER_WEIGHTS, args[1:1 + nw]))
    ad = args[1 + nw:1 + nw + 2 * len(targets)]
    lora = {t: (ad[2 * i], ad[2 * i + 1]) for i, t in enumerate(targets)}
    tabs = args[1 + nw + 2 * len(targets):5 + nw + 2 * len(targets)]
    rest = args[5 + nw + 2 * len(targets):]
    return h, w, lora, tabs, rest


def layer_fwd_fn(c, targets, scale):
    """(h, *weights, *adapters, cos, sin, mask, rot) -> h_out"""
    def fwd(*args):
        h, w, lora, tabs, _ = _split(c, targets, args)
        return L.layer(c, h, w, lora, scale, *tabs)
    return fwd


def layer_bwd_full_fn(c):
    """전체 파인튜닝: (h, *weights, cos, sin, mask, rot, g_out) -> (g_h, *g_weights)

    LoRA 없이 레이어 가중치 9종(LAYER_WEIGHTS 순서) 전부의 기울기를 낸다. 순전파를 다시 계산한다.
    """
    nw = len(L.LAYER_WEIGHTS)

    def bwd(*args):
        h, w, _, tabs, rest = _split(c, (), args)
        g_out = rest[0]

        def f(h_, *ws):
            return L.layer(c, h_, dict(zip(L.LAYER_WEIGHTS, ws)), {}, 1.0, *tabs)
        _, pull = torch.func.vjp(f, h, *[w[k] for k in L.LAYER_WEIGHTS])
        grads = pull(g_out)
        assert len(grads) == 1 + nw
        return tuple(grads)
    return bwd


def layer_bwd_full_acc_fn(c):
    """전체 파인튜닝 + 장치 누적: (h, *weights, cos, sin, mask, rot, g_out, *acc) -> (g_h, *(acc + g_weights))

    micro-batch 사이 기울기를 장치에서 더해 step 마다 한 번만 host 로 내린다. RNGD→host 가 3.1 GB/s 라
    (8B 기울기 16 GB = 5 s) 내리는 횟수가 학습 속도를 정한다.
    """
    nw = len(L.LAYER_WEIGHTS)
    full = layer_bwd_full_fn(c)

    def bwd(*args):
        acc = args[-nw:]
        grads = full(*args[:-nw])
        return (grads[0], *[a + g for a, g in zip(acc, grads[1:])])
    return bwd


def sqnorm_fn():
    """x[R, C] -> [R, 1] fp32 행별 제곱합(나머지 합은 host 에서). 전체를 한 번에 합치면 컴파일러가
    0차원 축소로 바꿔 "unsupported EDF node: Cpu(...ReduceByVe... shape [])" 로 거부한다.

    전체 파인튜닝에서 레이어 기울기의 크기를 장치에서 바로 구해, 큰 기울기를 다 내리기 전에 전역
    clip 계수를 정한다 — 레이어마다 내려받는 즉시 갱신을 시작할 수 있다.
    """
    return lambda x: x.float().pow(2).sum(dim=-1, keepdim=True)


def layer_bwd_fn(c, targets, scale):
    """(h, *weights, *adapters, cos, sin, mask, rot, g_out) -> (g_h, *g_adapters)

    순전파를 다시 계산하므로 레이어 단위 activation checkpointing 과 같다 — 장치에는
    레이어 입력 h 만 남기면 된다.
    """
    n_ad = 2 * len(targets)

    def bwd(*args):
        h, w, lora, tabs, rest = _split(c, targets, args)
        g_out = rest[0]
        flat = [x for t in targets for x in lora[t]]

        def f(h_, *ad):
            lo = {t: (ad[2 * i], ad[2 * i + 1]) for i, t in enumerate(targets)}
            return L.layer(c, h_, w, lo, scale, *tabs)
        _, pull = torch.func.vjp(f, h, *flat)
        grads = pull(g_out)
        assert len(grads) == 1 + n_ad
        return tuple(grads)
    return bwd


def head_rms_fn(c):
    """(h, norm) -> x = rms(h)"""
    return lambda h, norm: L.rms(h, norm, c.eps)


def head_rms_bwd_fn(c):
    """(h, norm, g_x[f32]) -> g_h"""
    def bwd(h, norm, gx):
        _, pull = torch.func.vjp(lambda h_: L.rms(h_, norm, c.eps).float(), h)
        return pull(gx)[0]
    return bwd


def ce_stats_fn():
    """(x, lm_j, bias_j, iota_j, labels) -> (m, s, t)"""
    return L.ce_chunk_stats


def ce_grad_fn():
    """(x, lm_j, bias_j, iota_j, labels, lse, w, acc) -> acc'"""
    return L.ce_chunk_grad


def head_fn(c):
    """(h, norm, lm, iota, labels, inv_n) -> (loss[1,1], g_h)"""
    def head(h, norm, lm, iota, labels, inv_n):
        loss, pull = torch.func.vjp(lambda h_: L.head_loss(c, h_, norm, lm, iota, labels, inv_n), h)
        (g_h,) = pull(torch.ones_like(loss))
        return loss, g_h
    return head
