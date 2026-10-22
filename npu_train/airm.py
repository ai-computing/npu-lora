"""ai-rm workload runner (rm-core protocol v0.1, entry point `airm.workloads` 이름 `npu-lora`).

기존 `npu-lora train` 을 rm-core 계약으로 감싸기만 한다 — 학습 경로는 cli.cmd_train 그대로다.

  from rm_core import Workload, registry
  run = registry.load("airm.workloads", "npu-lora", backend="rngd")
  res = run(Workload(kind="train", model="meta-llama/Llama-3.1-8B-Instruct"),
            data="/mnt/store/datasets/nsmc.jsonl", out="/mnt/store/adapters/nsmc", steps=400, accum=4)
  res["measurements"]          # [Measurement("tok_s", ..., tags={"workload": "train", ...})]

설정 키(생성자 기본값과 호출 인자 모두, 호출 인자가 우선) — 이름과 기본값은 `npu-lora train` 옵션과 같다:
  data, out (필수), backend (rngd | rngd:N | cuda[:N] | cpu, 기본 rngd), dataset (지표 태그용 이름),
  rank, alpha, targets, seq_len, steps, accum, lr, warmup, grad_clip, seed, full, save_every,
  merge_out, run_name, log_file, metrics_port, cache_dir, tactic, max_shapes, max_mm, kv_transport.
학습에는 KV 경로가 없으므로 kv_transport 는 비워 둬야 한다(주면 ContractError — 조용히 무시하지 않는다).
"""
from rm_core import ContractError, Measurement, Workload

from . import cli

BACKENDS = ("rngd", "furiosa", "cuda", "cpu")
KEYS = ("data", "out", "backend", "dataset", "rank", "alpha", "targets", "seq_len", "steps", "accum", "lr",
        "warmup", "grad_clip", "seed", "full", "save_every", "merge_out", "run_name", "log_file", "metrics_port",
        "cache_dir", "tactic", "max_shapes", "max_mm", "kv_transport")


class TrainRunner:
    name = "npu-lora"

    def __init__(self, **defaults):
        _check_keys(defaults)
        self.defaults = {"backend": "rngd", **defaults}

    def __call__(self, workload, **config):
        """학습을 끝까지 돌리고 {"out", "steps", "best_loss", "mode", "run", "measurements"} 를 돌려준다."""
        _check_keys(config)
        cfg = {**self.defaults, **config}
        if not isinstance(workload, Workload):
            raise ContractError(f"workload must be rm_core.Workload, got {type(workload).__name__}")
        workload.validate()
        if workload.kind != "train":
            raise ContractError(f"npu-lora runs kind='train' workloads, got {workload.kind!r}")
        if workload.precision != "bf16":
            raise ContractError(f"npu-lora trains in bf16 only, got precision={workload.precision!r}")
        if cfg.pop("kv_transport", ""):
            raise ContractError("npu-lora training has no KV transport; leave kv_transport empty")
        for k in ("data", "out"):
            if not cfg.get(k):
                raise ContractError(f"npu-lora: config key {k!r} is required")
        if cfg["backend"].partition(":")[0] not in BACKENDS:
            raise ContractError(f"unknown backend {cfg['backend']!r}; use rngd[:N], cuda[:N] or cpu")
        dataset = cfg.pop("dataset", None)

        argv = ["train", "--model", workload.model]
        for k, v in cfg.items():
            if v is None or v is False:
                continue
            flag = "--" + k.replace("_", "-")
            if isinstance(v, (list, tuple)):           # targets=("q", "v")
                v = ",".join(v)
            argv += [flag] if v is True else [flag, str(v)]
        a = cli.parser().parse_args(argv)
        res = a.fn(a)

        tags = {"component": "npu-lora", "workload": "train", "model": workload.model.rstrip("/").split("/")[-1],
                "vendor": cli.vendor(cfg["backend"]), "run_id": res["run"]}
        if dataset:
            tags["dataset"] = dataset
        tok_s = res["tokens"] / res["seconds"] if res["seconds"] else 0.0
        out = {k: res[k] for k in ("out", "steps", "best_loss", "mode", "run")}
        out["measurements"] = [Measurement("tok_s", round(tok_s, 1), tags).validate()]
        return out


def _check_keys(cfg):
    unknown = set(cfg) - set(KEYS)
    if unknown:
        raise ContractError(f"unknown npu-lora config keys {sorted(unknown)}; allowed {KEYS}")


# entry point 는 npu_train.airm:train. rm_core.registry.load 가 등록된 객체를 (설정 기본값 kwargs 로) 호출해
# 인스턴스를 만들므로, 클래스를 그 이름으로 내보낸다 — 로드 결과가 (Workload, **설정) 을 받는 runner 다.
train = TrainRunner
