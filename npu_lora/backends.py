"""실행 백엔드: 같은 학습 그래프를 RNGD / CPU / CUDA 에서 돌린다.

Backend.compile(key, gm, sample) 은 "장치 텐서를 받아 장치 텐서를 돌려주는 함수"를 준다.
furiosa 는 그래프를 AOT 컴파일하고 결과(EDF)를 캐시에 남긴다 — 같은 모양이면 다음 실행은 읽기만 한다.
"""
import hashlib
import importlib.metadata as md
import os
import sys
import threading
import time

import torch


class Backend:
    name = "base"

    def compile(self, key, gm, sample):
        raise NotImplementedError

    def to_device(self, t):
        return t

    def to_host(self, t, out=None):
        """장치 → host. out 을 주면 그 버퍼에 복사한다(매번 새로 할당하는 것보다 훨씬 빠르다)."""
        if out is None:
            return t
        out.copy_(t)
        return out

    def host_buffer(self, like):
        """to_host(out=)·업로드에 거듭 쓸 host 버퍼."""
        return torch.empty(like.shape, dtype=like.dtype)

    def sync(self):
        pass

    def describe(self):
        return {"backend": self.name}


class CPUBackend(Backend):
    """기준값. 그래프를 그대로 eager 로 돈다."""
    name = "cpu"

    def compile(self, key, gm, sample):
        return gm


class CUDABackend(Backend):
    name = "cuda"

    def __init__(self, index=0):
        self.dev = f"cuda:{index}"

    def compile(self, key, gm, sample):
        # make_fx 가 캡처한 상수 텐서(_tensor_constant*)는 buffer 가 아니라서 .to() 로 옮겨지지 않는다.
        for mod in gm.modules():
            for k, v in list(vars(mod).items()):
                if isinstance(v, torch.Tensor):
                    setattr(mod, k, v.to(self.dev))
        return gm.to(self.dev)

    def _copy_stream(self):
        # 레이어 작업 스레드마다 복사 전용 stream. 기본 stream 에서 돌면 복사가 계산 뒤에 줄을 선다.
        import threading
        local = self.__dict__.setdefault("_local", threading.local())
        if not hasattr(local, "stream"):
            local.stream = torch.cuda.Stream(self.dev)
        return local.stream

    def to_device(self, t):
        if t.device.type != "cpu":
            return t.to(self.dev)
        s = self._copy_stream()
        with torch.cuda.stream(s):
            out = (t if t.is_pinned() else t.pin_memory()).to(self.dev, non_blocking=True)
        s.synchronize()          # 돌려받은 텐서는 어느 stream 에서 써도 안전하다
        return out

    def host_buffer(self, like):
        return torch.empty(like.shape, dtype=like.dtype, pin_memory=True)

    def to_host(self, t, out=None):
        s = self._copy_stream()
        s.wait_stream(torch.cuda.current_stream(self.dev))   # t 를 만든 계산이 끝난 뒤 복사
        if out is None:
            out = self.host_buffer(t)
        with torch.cuda.stream(s):
            out.copy_(t, non_blocking=True)
        t.record_stream(s)
        s.synchronize()
        return out

    def sync(self):
        torch.cuda.synchronize(self.dev)

    def describe(self):
        return {"backend": self.name, "device": torch.cuda.get_device_name(self.dev)}


# furiosa-torch 2026.2 keeps the device memory type (DRAM / DRAM_SHARED) in one process-wide flag, and every
# compiled-graph call sets it to DRAM_SHARED for a moment. A device allocation from another thread in that moment
# lands in DRAM_SHARED, which has 0 bytes on a single-chip card: `OutOfMemoryError ... on DRAM_SHARED` (#40).
# The full-FT layer workers upload while the main thread runs graphs, so graph calls and device allocations share
# this lock. Only the allocation is under it; the host-to-device copy runs outside.
DEVICE_ALLOC_LOCK = threading.Lock()


class Serialized:
    """A compiled module whose calls hold `lock` (other attributes pass through)."""

    def __init__(self, fn, lock=DEVICE_ALLOC_LOCK):
        self.fn, self.lock = fn, lock

    def __call__(self, *args):
        with self.lock:
            return self.fn(*args)

    def __getattr__(self, name):
        return getattr(self.fn, name)


class FuriosaBackend(Backend):
    """FuriosaAI RNGD. furiosa-torch 의 CompileModule 로 그래프를 컴파일한다."""
    name = "furiosa"

    def __init__(self, chip=0, cache_dir=None, tactic="ForMiscModel", fusion=8, mesh=2, max_shapes=8,
                 compile_only=False):
        import furiosa.torch  # noqa: F401 — torch 다음, 다른 furiosa import 보다 먼저
        # 컴파일러는 ~/.cache/furiosa/compiler 에 내부 캐시를 쓰는데 디렉터리가 없으면 만들지 않고
        # "No such file or directory (os error 2)" 로 실패한다(새 컨테이너에서 첫 컴파일 때).
        os.makedirs(os.path.expanduser("~/.cache/furiosa/compiler"), exist_ok=True)
        # compile_only: 장치를 열지 않고 컴파일 캐시만 만든다. 컴파일은 CPU 작업이라 카드가 다른 작업에
        # 잡혀 있어도, 코어가 많은 다른 노드에서 미리 해 둘 수 있다(캐시 디렉터리를 복사해 쓴다).
        self.compile_only = compile_only
        if compile_only:
            chip = 0
        elif chip < 0:
            chip = visible_chip(-1 - chip)          # -1: first assigned card, -2: second, ...
        self.sdk = _version("furiosa-torch")
        self.new = tuple(int(x) for x in self.sdk.split(".")[:2] if x.isdigit()) >= (2026, 3)
        if self.new:
            from furiosa.torch import CompileModule, set_fusion, set_mesh_kind, MeshKind
            from furiosa.torch import current_stream
            self._stream = current_stream
            self.dev = f"furiosa:{chip}"
        else:
            from furiosa.torch.custom_ops import CompileModule
            from furiosa.torch import native_device as nd
            set_fusion, set_mesh_kind, MeshKind = nd.set_fusion, nd.set_mesh_kind, nd.MeshKind
            self._stream = None
            self.dev = f"rngd:{chip}"
        try:
            from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig
        except ImportError:
            from furiosa.torch._C.config.compiler import CompilerConfig as FCConfig, TacticHintConfig
        self.CompileModule = CompileModule
        # RNGD 의 fused node 는 PE 4개가 최대라, 8 PE 는 두 그룹을 mesh 로 묶는다.
        # 장치 텐서를 만들기 전에 불러야 한다.
        set_fusion(fusion)
        set_mesh_kind(MeshKind.Mesh(mesh))
        # max_num_unique_shapes 는 기본이 무제한이다. 8 로 묶으면 큰 그래프가 컴파일되기 시작하고
        # 커널도 빨라진다(rngd_pd 42f3e78: decode 26.2 → 17.6 ms). 0 이면 무제한.
        kw = {"max_num_unique_shapes": max_shapes} if max_shapes else {}
        self.cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                            tactic_hint=getattr(TacticHintConfig, tactic), **kw)
        self.max_shapes = max_shapes
        self.tactic = tactic
        self.chip = chip
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/npu-lora/furiosa")

    # -- 컴파일 캐시 -------------------------------------------------------------------------
    def _path(self, key):
        d = os.path.join(self.cache_dir, self.sdk)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, hashlib.blake2b(f"{key}|{self.tactic}|shapes{self.max_shapes}".encode(), digest_size=12).hexdigest())

    def _load(self, path):
        if self.new:
            return None                       # 2026.3+ 는 EDF 컨테이너가 없다 — 매번 컴파일
        from furiosa.native_torch import ir
        from furiosa.torch.custom_ops.edf import EdfModule
        with open(path + ".edf", "rb") as f:
            edf = ir.Edf.deserialize(f.read())
        return self.CompileModule(EdfModule(edf), torch.load(path + ".w", weights_only=False))

    def _save(self, m, path):
        if self.new:
            return
        with open(path + ".edf.tmp", "wb") as f:
            f.write(m.edf.serialize())
        os.replace(path + ".edf.tmp", path + ".edf")
        torch.save(m.module_weight, path + ".w.tmp")
        os.replace(path + ".w.tmp", path + ".w")

    def compile(self, key, gm, sample):
        from .graph import GraphModule
        path = self._path(key)
        t0 = time.time()
        m = None
        if os.path.exists(path + ".edf") and os.path.exists(path + ".w"):
            try:
                m = self._load(path)
                mode = "cache"
            except Exception as e:
                print(f"[furiosa] cache load failed for {key}: {e!r}; recompiling", flush=True)
        if m is None:
            m = self.CompileModule.from_module(GraphModule(gm), tuple(sample), compiler_config=self.cfg)
            mode = "compile"
            try:
                self._save(m, path)
            except Exception as e:
                print(f"[furiosa] could not cache {key}: {e!r}", flush=True)
        if self.compile_only:
            print(f"[furiosa] {key}: {mode} {time.time() - t0:.0f}s", flush=True)
            return m
        with DEVICE_ALLOC_LOCK:
            m.to(self.dev)
        print(f"[furiosa] {key}: {mode} {time.time() - t0:.0f}s", flush=True)
        return Serialized(m)

    def to_device(self, t):
        if self.compile_only:
            return t
        if t.device.type != "cpu":
            return t.to(self.dev)
        # allocate under the lock (see DEVICE_ALLOC_LOCK), copy outside it: same speed as t.to(dev), 6.0 GB/s
        with DEVICE_ALLOC_LOCK:
            out = torch.empty(t.shape, dtype=t.dtype, device=self.dev)
        return out.copy_(t)

    def to_host(self, t, out=None):
        # 미리 잡아 둔 버퍼로 받으면 19.9 GB/s, t.cpu() 는 매번 새 메모리를 잡느라 3.0 GB/s (RNGD)
        if out is None:
            return t.cpu()
        out.copy_(t)
        return out

    def sync(self):
        if self._stream is not None:
            self._stream(self.dev).synchronize()   # 2026.3+ 는 비동기 실행
        else:
            # 2026.2 는 동기 실행이지만, 장치→host 복사가 끝나야 결과가 확정된다.
            pass

    def describe(self):
        return {"backend": self.name, "device": self.dev, "sdk": self.sdk, "tactic": self.tactic,
                "max_shapes": self.max_shapes}


class RBLNBackend(Backend):
    """Rebellions ATOM. Each graph part is compiled with rebel.compile_from_torch and run by an RBLN runtime;
    inputs and outputs are host torch tensors (the runtime uploads and downloads per call). Compiled parts are
    cached as .rbln files keyed by graph key, sample shapes/dtypes and the compiler version.

    rebel-compiler 0.11 rejects neg + add.Scalar (the SiLU backward; it segfaults when a matmul follows), so
    graph.fold_neg_add rewrites it to rsub first. Each part still compiles in a child process, and a part that
    does not compile runs eagerly on the host (recorded in `fallbacks` and in describe()). Use --max-mm 2 so a
    failure costs one small part.

    Numerics: ATOM rounds results away from zero far more often than round-to-nearest-even (bias +0.5..0.9 of
    the error, experiments/data/rbln-atom/repro_rounding.py), and fp32 graph outputs carry bf16-level error. With
    every part on ATOM the forward activations that leave a layer drift, the head's softmax amplifies that, and
    `npu-lora check` (2 layers of Llama-3.2-1B) is 2-13 % off the CPU. `host_parts` places whole part groups on
    the host on purpose; the default ("fwd") runs the layer forward there and the backward (which recomputes
    each layer's forward from its input) and the head on ATOM. host_parts=() puts everything on ATOM."""
    name = "rbln"

    # graph-part groups the trainer compiles (the key suffix before -p<i>of<n>-<hash>); the longest name matches first
    GROUPS = ("rms-bwd", "bwd-acc", "ce-stats", "ce-grad", "fwd", "bwd", "rms")

    def __init__(self, device=0, cache_dir=None, host_fallback=True, host_parts=None):
        import rebel
        self.rebel = rebel
        self.dev = device
        self.sdk = getattr(rebel, "__version__", None) or _version("rebel-compiler")
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/npu-lora/rbln")
        self.host_fallback = host_fallback
        # groups placed on the host on purpose. Default: the layer forward (see the class docstring)
        self.host_parts = ("fwd",) if host_parts is None else tuple(host_parts)
        self.fallbacks = []            # (key, op names) of graph parts the compiler rejected (run on the host)
        self.placed = []               # keys placed on the host by host_parts
        self.seconds = {"device": 0.0, "host": 0.0}   # wall time in graph-part calls, by where they ran

    @classmethod
    def group(cls, key):
        """'llama-...-s2.0-bwd-p16of39-f460c97c8d2f' -> 'bwd' (None for keys of other graphs)"""
        import re
        m = re.search(r"-(%s)-p\d+of\d+-[0-9a-f]+$" % "|".join(cls.GROUPS), key)
        return m.group(1) if m else None

    def _timed(self, where, fn):
        def run(*args):
            t0 = time.perf_counter()
            y = fn(*args)
            self.seconds[where] += time.perf_counter() - t0
            return y
        return run

    def _path(self, key, sample, code=""):
        sig = "|".join(f"{tuple(t.shape)}:{t.dtype}" for t in sample if isinstance(t, torch.Tensor))
        d = os.path.join(self.cache_dir, self.sdk)
        os.makedirs(d, exist_ok=True)
        # the graph text after the backend's own rewrites: a changed rewrite must not reuse an old .rbln/.host
        return os.path.join(d, hashlib.blake2b(f"{key}|{sig}|{code}".encode(), digest_size=12).hexdigest() + ".rbln")

    def compile(self, key, gm, sample):
        import copy
        from . import graph as G
        t0 = time.time()
        if self.group(key) in self.host_parts:
            self.placed.append(key)
            return self._timed("host", gm)
        gm = copy.deepcopy(gm)
        G.fold_neg_add(gm)              # rebel-compiler 0.11 rejects neg + add.Scalar (the SiLU backward)
        # the RBLN compiler drops graph inputs a part does not use, so its runtime takes fewer arguments than
        # the part: compile without them and pass only the used ones
        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        used = [i for i, n in enumerate(phs) if n.users]
        if len(used) < len(phs):
            for n in [n for n in gm.graph.nodes if n.op == "placeholder" and not n.users]:
                gm.graph.erase_node(n)
            gm.recompile()
            sample = [sample[i] for i in used]
            inner = self._compile_used(key, gm, sample, t0)
            return lambda *args: inner(*[args[i] for i in used])
        return self._compile_used(key, gm, sample, t0)

    def _compile_used(self, key, gm, sample, t0):
        path = self._path(key, sample, gm.code)
        if os.path.exists(path + ".host"):
            return self._host(key, gm)
        if os.path.exists(path):
            compiled, mode = self.rebel.RBLNCompiledModel(path), "cache"
        else:
            names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
            info = [(n, list(t.shape), str(t.dtype).replace("torch.", "")) for n, t in zip(names, sample)]
            err = self._compile_isolated(gm, info, path)    # the compiler can crash the process (segfault)
            if err is not None:
                if not self.host_fallback:
                    raise RuntimeError(f"RBLN compile of {key} failed: {err}")
                open(path + ".host", "w").write(err + "\n")
                return self._host(key, gm)
            compiled, mode = self.rebel.RBLNCompiledModel(path), "compile"
        rt = compiled.create_runtime(device=self.dev, tensor_type="pt")
        # the compiler may also drop inputs it folded away: map the part's arguments by name to what it kept
        names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        index = getattr(rt, "_input_name_to_index", None) or {}
        order = sorted((index[n], i) for i, n in enumerate(names) if n in index)
        pick = [i for _, i in order] if order and len(order) < len(names) else None
        print(f"[rbln] {key}: {mode} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        n_out = len(next(n for n in gm.graph.nodes if n.op == "output").args[0]) \
            if isinstance(next(n for n in gm.graph.nodes if n.op == "output").args[0], (tuple, list)) else 1

        def run(*args):
            if pick is not None:
                args = [args[i] for i in pick]
            y = rt(*[a.contiguous() for a in args])
            if isinstance(y, (list, tuple)):
                return tuple(y) if n_out > 1 else y[0]
            return y
        return self._timed("device", run)

    def _compile_isolated(self, gm, info, path):
        """compile in a child process; None on success (the .rbln is written), else the error text."""
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gm.pt")
            torch.save({"gm": gm, "info": info}, src)
            code = ("import os, sys, torch, rebel\n"
                    "o = torch.load(sys.argv[1], weights_only=False)\n"
                    "c = rebel.compile_from_torch(o['gm'], o['info'])\n"
                    "c.save(sys.argv[2] + '.tmp'); os.replace(sys.argv[2] + '.tmp', sys.argv[2])\n")
            r = subprocess.run([sys.executable, "-c", code, src, path], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(path):
            return None
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return f"exit {r.returncode}: " + " | ".join(tail)[:500]

    def _host(self, key, gm):
        """a part the RBLN compiler rejected runs eagerly on the host (functional fallback, recorded)"""
        ops = sorted({str(n.target).replace("aten.", "") for n in gm.graph.nodes if n.op == "call_function"})
        self.fallbacks.append((key, ops))
        print(f"[rbln] {key}: host fallback (ops: {', '.join(ops)})", file=sys.stderr, flush=True)
        return self._timed("host", gm)

    def describe(self):
        return {"backend": self.name, "device": f"rbln{self.dev}", "sdk": self.sdk,
                "host_fallback_parts": len(self.fallbacks), "host_parts": list(self.host_parts),
                "host_placed_parts": len(self.placed)}


def visible_chip(nth=0):
    """쓸 RNGD 의 furiosa-torch 번호.

    furiosa-torch 의 `rngd:N` 은 /dev/rngd 에 보이는 카드 중 N 번째(서수)다 — npu-node-1 처럼 npu0·npu1·npu3
    이 보이면 rngd:2 가 npu3 이다. 또 furiosa-torch 는 privileged 가 아니면 장치를 열지 못한다(device plugin
    이 노출한 카드만으로는 `device index out of range`). 그래서 쿠버네티스에서는 privileged 가 아닌 init
    컨테이너가 배정받은 카드 이름(npuN)을 NPU_LORA_RNGD_FILE 에 적고, 여기서 그 카드의 서수를 찾는다.
    둘 다 없으면 첫 카드. A multi-card pod lists several names ("npu1,npu3"); nth picks one of them in order.
    """
    import re
    try:
        ids = sorted({int(m.group(1)) for f in os.listdir("/dev/rngd") if (m := re.match(r"npu(\d+)", f))})
    except FileNotFoundError:
        ids = []
    if not ids:
        raise RuntimeError("no RNGD visible under /dev/rngd")
    want = os.environ.get("NPU_LORA_RNGD", "")
    path = os.environ.get("NPU_LORA_RNGD_FILE", "")
    if not want and path and os.path.exists(path):
        want = open(path).read().strip()
    if want:
        names = [x for x in re.split(r"[,\s]+", want) if x]
        if nth >= len(names):
            raise RuntimeError(f"card {nth} requested but only {names} are assigned")
        n = int(re.sub(r"\D", "", names[nth]))
        if n not in ids:
            raise RuntimeError(f"assigned RNGD {names[nth]} is not visible ({ids}); is the container privileged?")
        return ids.index(n)
    return nth


def _version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "unknown"


def make(spec, nth=0, **kw):
    """'furiosa:0' / 'furiosa:auto'(컨테이너에 보이는 카드) / 'cuda:0' / 'cpu' → Backend.
    nth: for 'furiosa:auto', which of the pod's assigned cards (`--backend furiosa:auto,furiosa:auto` = both)."""
    kind, _, idx = spec.partition(":")
    if kind in ("furiosa", "rngd"):
        return FuriosaBackend(chip=-1 - nth if idx in ("", "auto") else int(idx), **kw)
    if kind == "cuda":
        return CUDABackend(int(idx or 0))
    if kind == "cpu":
        return CPUBackend()
    if kind in ("rbln", "atom"):
        return RBLNBackend(int(idx or 0), cache_dir=kw.get("cache_dir"), host_parts=kw.get("host_parts"))
    raise ValueError(f"unknown backend {spec!r} (furiosa:N, rbln:N, cuda:N, cpu)")
