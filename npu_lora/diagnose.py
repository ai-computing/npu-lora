"""컴파일이 실패하는 그래프에서 처음 실패하는 노드를 찾는다.

그래프를 앞에서부터 k 번째 노드까지만 남기고(그 노드를 출력으로) 컴파일해 보며 이분 탐색한다.
새 모델 구조나 새 SDK 에서 컴파일러가 받지 못하는 패턴을 찾을 때 쓴다 — 찾은 패턴은 graph.PASSES 에
의미를 바꾸지 않는 변환으로 넣는다.
"""
import copy
import hashlib
import time


def _prefix(gm, k):
    g = copy.deepcopy(gm)
    nodes = [n for n in g.graph.nodes if n.op == "call_function"]
    out = [n for n in g.graph.nodes if n.op == "output"][0]
    out.args = ((nodes[k],),)
    g.graph.eliminate_dead_code()
    g.recompile()
    return g


def first_failing_node(backend, gm, sample, log=print):
    """(index, node) — 그래프 전체가 컴파일되면 None."""
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]

    def depends_on_input(k):
        seen, stack = set(), [nodes[k]]
        while stack:
            x = stack.pop()
            if x.op == "placeholder":
                return True
            for a in x.all_input_nodes:
                if a not in seen:
                    seen.add(a)
                    stack.append(a)
        return False

    def ok(k):
        if not depends_on_input(k):        # 상수만 내는 앞부분은 그 자체로 컴파일이 안 될 수 있다 — 판단에서 뺀다
            return True
        t0 = time.time()
        try:
            backend.compile(f"diagnose-{id(gm)}-{k}-{time.time()}", _prefix(gm, k), sample)
            res = True
        except BaseException:
            res = False
        log(f"[diagnose] prefix {k} ({nodes[k].target}): {'ok' if res else 'FAIL'} {time.time() - t0:.0f}s")
        return res

    if ok(len(nodes) - 1):
        return None
    lo, hi = 0, len(nodes) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo, nodes[lo]


def compare_parts(backend, key, gm, sample, max_mm=2, log=print):
    """조각마다 CPU 출력과 장치 출력을 같은 입력으로 비교한다 — 틀린 값을 내는 조각 찾기.

    컴파일은 성공했는데 결과가 틀릴 때(출력 순서, 레이아웃, 융합 버그) 어느 조각인지 좁힌다.
    반환: [{part, ops, outputs: [rel_err ...]}]
    """
    import torch
    from torch.fx.passes.split_module import split_module
    from . import graph as G
    assign = G.plan_parts(gm, max_mm)
    split = split_module(gm, None, lambda n: assign[n], keep_original_order=True)
    names = [n for n, _ in split.named_children() if n.startswith("submod_")]
    io = {}
    hooks = [getattr(split, n).register_forward_hook(lambda m, a, o, n=n: io.__setitem__(n, (a, o)))
             for n in names]
    with torch.no_grad():
        split(*sample)
    for h in hooks:
        h.remove()
    rows = []
    for i, n in enumerate(names):
        sub = getattr(split, n)
        args, ref = io[n]
        args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
        G.materialize_outputs(sub, ref)
        sig = sub.code + repr([(tuple(a.shape), str(a.dtype), a.stride()) for a in args if isinstance(a, torch.Tensor)])
        code_hash = hashlib.blake2b(sig.encode(), digest_size=6).hexdigest()
        fn = backend.compile(f"{key}-p{i}of{len(names)}-{code_hash}", sub, args)
        out = fn(*[backend.to_device(a) for a in args])
        backend.sync()
        out = out if isinstance(out, (tuple, list)) else (out,)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        errs = []
        for o, r in zip(out, ref):
            o, r = backend.to_host(o).float(), r.float()
            errs.append(float((o - r).norm() / (r.norm() + 1e-12)) if o.shape == r.shape else f"shape {tuple(o.shape)} vs {tuple(r.shape)}")
        ops = [str(x.target).replace("aten.", "") for x in sub.graph.nodes if x.op == "call_function"]
        rows.append({"part": i, "n_out": len(ref), "errs": errs, "ops": " ".join(ops)})
        log(f"[compare] part {i}: outs={len(ref)} errs={['%.3g' % e if isinstance(e, float) else e for e in errs]}")
    return rows


def describe(node):
    def shp(a):
        v = getattr(a, "meta", {}).get("val")
        return (tuple(v.shape), str(v.dtype).replace("torch.", ""), tuple(v.stride())) if v is not None else repr(a)
    return {"node": node.name, "op": str(node.target), "args": [shp(a) for a in node.args],
            "out": shp(node)}
