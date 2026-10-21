"""npu-lora 명령.

  npu-lora check  --model M --backend furiosa:0 [--layers 2]      # NPU 기울기 = CPU 기울기?
  npu-lora train  --model M --data d.jsonl --backend furiosa:0 --out DIR [--steps N ...]
  npu-lora merge  --model M --adapter DIR --out DIR               # NPU 서빙용 합친 모델
"""
import argparse
import json
import os
import socket
import sys
import time


def _common(p):
    p.add_argument("--model", required=True, help="HF 이름(로컬 캐시) 또는 경로")
    p.add_argument("--backend", default="furiosa:0", help="furiosa:N | cuda:N | cpu")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--targets", default="q,v", help="LoRA 대상 투영: q,k,v,o 중")
    p.add_argument("--tactic", default="ForMiscModel", help="furiosa tactic hint")
    p.add_argument("--cache-dir", default=None, help="컴파일 캐시 (기본 ~/.cache/npu-lora)")
    p.add_argument("--max-shapes", type=int, default=8, help="furiosa max_num_unique_shapes (0=무제한)")
    p.add_argument("--max-mm", type=int, default=None,
                   help="그래프 조각당 행렬곱 수 (기본: furiosa 2, 그 외 0=통째로)")


def _backend(a):
    """--backend furiosa:0 → one backend; a comma list (furiosa:0,furiosa:1) → a list (full FT layer split)."""
    from . import backends
    kw = {"tactic": a.tactic, "cache_dir": a.cache_dir, "max_shapes": a.max_shapes} if a.backend.startswith(("furiosa", "rngd")) else {}
    specs = a.backend.split(",")
    bes = [backends.make(x, nth=i, **kw) for i, x in enumerate(specs)]   # furiosa:auto,furiosa:auto = the pod's two cards
    return bes if len(bes) > 1 else bes[0]


def vendor(backend):
    """백엔드 지정 → ai-rm vendor 이름(rm_core.accelerator.VENDORS)."""
    return {"furiosa": "furiosa", "rngd": "furiosa", "cuda": "nvidia", "cpu": "cpu"}.get(backend.partition(":")[0], "other")


def _lora(a):
    from .trainer import LoRAConfig
    return LoRAConfig(r=a.rank, alpha=a.alpha, targets=tuple(a.targets.split(",")))


def cmd_train(a):
    from transformers import AutoTokenizer
    from . import export
    from .data import PackedDataset
    from .metrics import Reporter
    from .trainer import LoRATrainer, TrainConfig

    run = a.run_name or f"lora-{time.strftime('%Y%m%d-%H%M%S')}"
    model = os.path.basename(a.model.rstrip("/"))
    rep = Reporter(run, labels={"model": model, "backend": a.backend, "host": socket.gethostname()},
                   log_file=a.log_file, port=a.metrics_port,
                   airm_tags={"component": "npu-lora", "workload": "train", "model": model,
                              "vendor": vendor(a.backend), "run_id": run})
    tc = TrainConfig(seq_len=a.seq_len, lr=a.lr, steps=a.steps, accum=a.accum, warmup=a.warmup,
                     grad_clip=a.grad_clip, seed=a.seed, full=a.full, resident_layers=a.resident_layers,
                     update_workers=a.update_workers)
    tr = LoRATrainer(a.model, _backend(a), _lora(a), tc, log=lambda s: rep.event("log", **json.loads(s)),
                     max_mm=a.max_mm)
    tok = AutoTokenizer.from_pretrained(tr.path)
    ds = PackedDataset(a.data, tok, a.seq_len, seed=a.seed)
    rep.event("start", examples=len(ds), truncated=ds.truncated, **tr.summary())
    tr.setup()
    rep.set("npu_lora_steps_planned", a.steps)
    best = None
    tokens, seconds = 0, 0.0
    for batch in ds.batches(a.accum, a.steps):
        m = tr.train_step(batch)
        rep.step(m)
        best = m["loss"] if best is None else min(best, m["loss"])
        tokens += m["tokens"]
        seconds += m["seconds"]
        if a.save_every and m["step"] % a.save_every == 0 and not a.full:
            export.save_peft(tr, os.path.join(a.out, f"step-{m['step']}"), a.model)
    extra = {"final_step": tr.step_no, "best_loss": best, "run": run}
    if a.full:
        export.save_full(tr, a.out, extra=extra)
    else:
        export.save_peft(tr, a.out, a.model, extra=extra)
    done = {"out": a.out, "steps": tr.step_no, "best_loss": best, "mode": "full" if a.full else "lora"}
    rep.event("done", **done)
    if a.merge_out and not a.full:
        export.save_merged(a.model, a.out, a.merge_out)
        rep.event("merged", out=a.merge_out)
    return {**done, "run": run, "tokens": tokens, "seconds": seconds}


def cmd_check(a):
    """같은 입력·어댑터로 NPU 와 CPU 의 loss·기울기를 비교한다(앞 N 개 레이어만 써서 빠르게)."""
    import torch
    from . import backends
    from .trainer import LoRATrainer, TrainConfig
    from .moe import RoutingPins
    tc = TrainConfig(seq_len=a.seq_len, full=a.full)
    torch.manual_seed(0)
    res, pins = {}, {}
    for name, be in (("npu", _backend(a)), ("cpu", backends.CPUBackend())):
        tr = LoRATrainer(a.model, be, _lora(a), tc, log=lambda s: print(s, file=sys.stderr),
                         max_mm=a.max_mm if name == "npu" else 0)
        tr.c = tr.c.__class__(**{**tr.c.__dict__, "n_layers": a.layers})   # 앞 N 개 레이어만
        tr.setup()
        g = torch.Generator().manual_seed(1)
        if not a.full:
            for per in tr.master:                     # B=0 이면 A 기울기가 0 — 비교가 안 된다
                for b in per[1::2]:
                    b.detach().normal_(0, 0.01, generator=g)
        tr._upload_adapters()
        ids = torch.randint(0, tr.c.vocab, (a.seq_len,), generator=g)
        labels = torch.roll(ids, -1)
        t0 = time.time()
        if tr.host_mlp is None:
            loss, grads, _ = tr.forward_backward(ids, labels)
        else:       # MoE: the CPU reference uses the experts the NPU run picked (see moe.RoutingPins)
            pins[name] = RoutingPins(tr.host_mlp.blocks)
            if name == "cpu" and not a.free_routing:
                with RoutingPins(tr.host_mlp.blocks) as free:          # unpinned pass: how many tokens flip
                    free.mode = "record"
                    tr.forward_backward(ids, labels)
                pins["free"] = free
                pins[name].ids, pins[name].mode = pins["npu"].ids, "pin"
            else:
                pins[name].mode = "record"
            with pins[name]:
                loss, grads, _ = tr.forward_backward(ids, labels)
        res[name] = (loss, grads, time.time() - t0)
        print(json.dumps({"backend": name, "loss": loss, "seconds": round(res[name][2], 2)}), file=sys.stderr)
    (ln, gn, tn), (lc, gc, tcpu) = res["npu"], res["cpu"]
    worst = 0.0
    rows = []
    for i, (x, y) in enumerate(zip(gn, gc)):
        for j, (p, q) in enumerate(zip(x, y)):
            rel = float((p - q).norm() / (q.norm() + 1e-12))
            cos = float(torch.nn.functional.cosine_similarity(p.flatten(), q.flatten(), dim=0))
            worst = max(worst, rel)
            rows.append({"layer": i, "tensor": j, "rel_err": round(rel, 5), "cos": round(cos, 6)})
    out = {"loss_npu": ln, "loss_cpu": lc, "loss_rel_err": abs(ln - lc) / abs(lc), "grad_worst_rel_err": worst,
           "npu_seconds": round(tn, 2), "cpu_seconds": round(tcpu, 2), "grads": rows,
           **({"routing": "pinned to the NPU run's experts" if "free" in pins else "free",
               "routing_flips_unpinned": pins["npu"].flips(pins["free"]) if "free" in pins else None,
               "tokens": a.seq_len} if pins else {}),
           "pass": abs(ln - lc) / abs(lc) < 1e-2 and worst < 5e-2}
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)


def cmd_compile(a):
    """장치 없이 학습 그래프를 컴파일해 캐시에 넣는다(코어 많은 노드에서 미리)."""
    from . import backends
    from .trainer import LoRATrainer, TrainConfig
    be = backends.FuriosaBackend(chip=0, cache_dir=a.cache_dir, tactic=a.tactic, max_shapes=a.max_shapes,
                                 compile_only=True)
    if a.adamw_only:
        _compile_adamw(a, be)
        return
    tc = TrainConfig(seq_len=a.seq_len, accum=a.accum, full=a.full, resident_layers=a.resident_layers)
    tr = LoRATrainer(a.model, be, _lora(a), tc, log=lambda s: print(s, flush=True), max_mm=a.max_mm)
    t0 = time.time()
    tr.setup()
    print(json.dumps({"event": "compiled", "seconds": round(time.time() - t0, 1), "cache": be.cache_dir}))


def _compile_adamw(a, be):
    """Only the resident device AdamW graphs (`train --full --resident-layers N`): one per 2-D weight shape.

    They depend on the weight shapes alone, so this reads no weights and builds no training graph. The full
    `compile` path first re-traces and re-runs every training graph part on the CPU (bf16, even when every
    part is already cached); on a shared CPU node that did not reach the AdamW graphs in 2.5 h (#34).
    """
    from . import llama as L
    from .optim_device import ResidentAdamW, resident_shapes
    path = L.resolve_model_path(a.model)
    c = L.LlamaConfig.from_hf(path)
    ra = ResidentAdamW(be, log=lambda m: print(m, flush=True))
    t0 = time.time()
    for shape in resident_shapes(L.Weights(path), c):
        t = time.time()
        ra.compile(shape)
        print(json.dumps({"event": "compiled_adamw", "shape": list(shape), "seconds": round(time.time() - t, 1)}), flush=True)
    print(json.dumps({"event": "compiled", "seconds": round(time.time() - t0, 1), "cache": be.cache_dir}))


def cmd_merge(a):
    from . import export
    export.save_merged(a.model, a.adapter, a.out)
    print(json.dumps({"event": "merged", "out": a.out}))


def cmd_serve(a):
    """학습한 어댑터를 실행 중인 vLLM(--enable-lora, VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)에 올린다.

    vLLM 이 어댑터 경로를 직접 읽으므로 path 는 vLLM 컨테이너 안 경로다(보통 공유 스토어
    /mnt/store/adapters/...). 올린 뒤 /v1/models 에 이름이 보이면 성공 — 요청의 "model" 을 그 이름으로.
    """
    import urllib.request

    def call(path, body=None):
        req = urllib.request.Request(a.endpoint.rstrip("/") + path,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode()

    if a.unload:
        print(call("/v1/unload_lora_adapter", {"lora_name": a.name}))
        return
    print(call("/v1/load_lora_adapter", {"lora_name": a.name, "lora_path": a.path}))
    models = [m["id"] for m in json.loads(call("/v1/models"))["data"]]
    print(json.dumps({"event": "served", "name": a.name, "models": models, "ok": a.name in models}))
    sys.exit(0 if a.name in models else 1)


def cmd_eval(a):
    """평가 JSONL({"prompt","expect"})을 OpenAI 호환 서버에 묻고, 답에 expect 가 들어 있는 비율."""
    import urllib.request
    rows, hit = [], 0
    for line in open(a.data):
        ex = json.loads(line)
        body = {"model": a.model, "messages": [{"role": "user", "content": ex["prompt"]}],
                "max_tokens": a.max_tokens, "temperature": 0}
        req = urllib.request.Request(a.endpoint.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        ans = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
        ok = ex["expect"].lower() in ans.lower()
        hit += ok
        rows.append({"prompt": ex["prompt"], "expect": ex["expect"], "ok": ok, "answer": ans.strip()[:200]})
    out = {"model": a.model, "accuracy": hit / max(len(rows), 1), "n": len(rows), "rows": rows}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_job(a):
    from . import k8s
    docs = k8s.build(a.model, a.data, a.name, ns=a.namespace, steps=a.steps, seq_len=a.seq_len, rank=a.rank,
                     alpha=a.alpha, targets=a.targets, lr=a.lr, accum=a.accum, image=a.image, node=a.node,
                     hf_host_path=a.hf_host_path, cache_host_path=a.cache_host_path, merge=a.merge)
    print(k8s.to_yaml(docs))


def parser():
    ap = argparse.ArgumentParser(prog="npu-lora", description="LoRA and full fine-tuning on AOT-compiled NPUs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="LoRA 또는 --full 전체 파인튜닝")
    _common(t)
    t.add_argument("--data", required=True, help="JSONL: text | prompt/response | messages")
    t.add_argument("--out", required=True, help="PEFT 어댑터 출력 디렉터리")
    t.add_argument("--steps", type=int, default=100)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save-every", type=int, default=0)
    t.add_argument("--merge-out", default=None, help="끝나면 합친 모델도 쓴다(NPU 서빙용)")
    t.add_argument("--full", action="store_true",
                   help="전체 파인튜닝: 레이어 가중치 전부 학습(--out 에 HF 전체 모델). 임베딩·lm_head 는 고정")
    t.add_argument("--resident-layers", type=int, default=0,
                   help="--full: per device, the first N layers keep fp32 master + AdamW state on the device")
    t.add_argument("--update-workers", type=int, default=4,
                   help="--full: host threads for the per-layer offload jobs (gradient D2H, AdamW, H2D)")
    t.add_argument("--run-name", default=None)
    t.add_argument("--log-file", default=None)
    t.add_argument("--metrics-port", type=int, default=None, help="Prometheus /metrics 포트")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("check", help="NPU 기울기를 CPU 기준값과 비교")
    _common(c)
    c.add_argument("--layers", type=int, default=2)
    c.add_argument("--full", action="store_true", help="전체 파인튜닝 기울기(레이어 가중치)를 비교")
    c.add_argument("--free-routing", action="store_true",
                   help="MoE: CPU 기준이 자기 라우팅을 쓴다(기본: NPU 실행이 고른 expert 로 고정해 수치 오차만 비교)")
    c.set_defaults(fn=cmd_check)

    cc = sub.add_parser("compile", help="장치 없이 학습 그래프를 미리 컴파일(캐시만 채움)")
    _common(cc)
    cc.add_argument("--full", action="store_true")
    cc.add_argument("--accum", type=int, default=1)
    cc.add_argument("--resident-layers", type=int, default=0, help="also compile the resident device AdamW graphs")
    cc.add_argument("--adamw-only", action="store_true",
                    help="compile only the resident device AdamW graphs (no weights loaded, no training graphs)")
    cc.set_defaults(fn=cmd_compile)

    m = sub.add_parser("merge", help="어댑터를 base 가중치에 합친다")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)

    sv = sub.add_parser("serve", help="어댑터를 실행 중인 vLLM 에 올린다(재시작 없음)")
    sv.add_argument("--endpoint", required=True, help="vLLM 주소, 예 http://<vllm-host>:8000")
    sv.add_argument("--name", required=True, help="서빙 이름 = 요청의 model 값")
    sv.add_argument("--path", default=None, help="vLLM 컨테이너 안 어댑터 경로")
    sv.add_argument("--unload", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="서빙 중인 모델에 평가 질문을 묻고 핵심어 적중률을 잰다")
    ev.add_argument("--endpoint", required=True)
    ev.add_argument("--model", required=True, help="요청의 model 값(어댑터 이름 또는 base)")
    ev.add_argument("--data", required=True, help='JSONL: {"prompt", "expect"}')
    ev.add_argument("--max-tokens", type=int, default=96)
    ev.set_defaults(fn=cmd_eval)

    j = sub.add_parser("job", help="쿠버네티스 학습 Job 매니페스트 출력 (kubectl apply -f -)")
    j.add_argument("--model", required=True)
    j.add_argument("--data", required=True, help="컨테이너 안 경로(보통 /mnt/store/datasets/...)")
    j.add_argument("--name", required=True, help="실행 이름 — Job·출력 디렉터리 이름이 된다")
    j.add_argument("--namespace", default="llm-serving")
    j.add_argument("--steps", type=int, default=100)
    j.add_argument("--seq-len", type=int, default=512)
    j.add_argument("--rank", type=int, default=16)
    j.add_argument("--alpha", type=float, default=32.0)
    j.add_argument("--targets", default="q,v")
    j.add_argument("--lr", type=float, default=2e-4)
    j.add_argument("--accum", type=int, default=1)
    j.add_argument("--image", default="furiosaai/furiosa-llm:latest")
    j.add_argument("--node", default=None, help="노드 고정(kubernetes.io/hostname)")
    j.add_argument("--hf-host-path", default=None, help="HF 캐시를 노드 hostPath 에서 읽기")
    j.add_argument("--cache-host-path", default=None, help="컴파일 캐시를 노드 hostPath 에 두기")
    j.add_argument("--merge", action="store_true", help="합친 모델도 출력(NPU 서빙용)")
    j.set_defaults(fn=cmd_job)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
