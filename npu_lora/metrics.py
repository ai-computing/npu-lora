"""학습 진행을 바깥에서 읽는 두 경로.

1. JSONL: stdout 과 --log-file 에 한 줄씩(이벤트·step 지표). kubectl logs 로 그대로 읽힌다.
2. Prometheus: --metrics-port 를 주면 /metrics 를 연다. ServiceMonitor 로 기존 Prometheus 가
   수집하고 lmd-top 이 같은 소스를 읽는다. 이름은 npu_lora_* 로 시작한다. ai-rm 공통 지표
   (rm_core.metrics.METRICS)에 있는 양은 그 이름으로만 낸다: 학습 처리량은 airm_tok_s 이고,
   라벨은 rm-core TAG_KEYS 만 쓴다(airm_tags). 이 모듈은 학습 Job 이미지에서 돌므로 rm_core 를 import 하지 않는다.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Reporter:
    def __init__(self, run_name, labels=None, log_file=None, port=None, airm_tags=None):
        self.run = run_name
        self.labels = {"run": run_name, **(labels or {})}
        self.airm_tags = airm_tags or {}
        self.airm = {}
        self.f = open(log_file, "a") if log_file else None
        self.gauges = {}
        self.counters = {}
        self.lock = threading.Lock()
        if port:
            self._serve(port)

    def event(self, kind, **kw):
        rec = {"ts": round(time.time(), 3), "run": self.run, "event": kind, **kw}
        line = json.dumps(rec, default=str)
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def step(self, m):
        self.event("step", **m)
        with self.lock:
            self.gauges.update({"npu_lora_loss": m["loss"], "npu_lora_grad_norm": m["grad_norm"],
                                "npu_lora_learning_rate": m["lr"], "npu_lora_step": m["step"],
                                "npu_lora_step_seconds": m["seconds"]})
            self.airm["airm_tok_s"] = m["tokens_per_s"]
            self.counters["npu_lora_tokens_total"] = self.counters.get("npu_lora_tokens_total", 0) + m["tokens"]

    def set(self, name, value):
        with self.lock:
            self.gauges[name] = value

    def _render(self):
        lab = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        out = []
        with self.lock:
            for k, v in sorted(self.gauges.items()):
                out += [f"# TYPE {k} gauge", f"{k}{{{lab}}} {v}"]
            for k, v in sorted(self.counters.items()):
                out += [f"# TYPE {k} counter", f"{k}{{{lab}}} {v}"]
            alab = ",".join(f'{k}="{v}"' for k, v in self.airm_tags.items())
            for k, v in sorted(self.airm.items()):
                out += [f"# TYPE {k} gauge", f"{k}{{{alab}}} {v}"]
        return "\n".join(out) + "\n"

    def _serve(self, port):
        rep = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") not in ("/metrics", "/healthz"):
                    self.send_response(404)
                    self.end_headers()
                    return
                body = (rep._render() if self.path.startswith("/metrics") else "ok\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(json.dumps({"event": "metrics_listening", "port": port}), file=sys.stderr, flush=True)
