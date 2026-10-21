"""Training energy from Prometheus raw samples (stdlib only).

usage: power.py LOG.jsonl NODE OUT_PREFIX [--skip N] [--idle-before SECONDS] [--prom URL]
LOG: npu_lora training stdout (JSON lines with "event": "step" and "ts").
Steady-state window: from the end of step N (default 5) to the end of the last step.
Writes OUT_PREFIX.power.json (summary) and OUT_PREFIX.power.csv (raw samples).
"""
import csv
import json
import statistics
import sys
import urllib.parse
import urllib.request

args = sys.argv[1:]
log, node, out = args[:3]
opt = dict(zip(args[3::2], args[4::2]))
skip = int(opt.get("--skip", 5))
idle_before = float(opt.get("--idle-before", 0))
idle_at = opt.get("--idle-at")                    # explicit idle window end (unix ts) instead of before the run
prom = opt.get("--prom", "http://127.0.0.1:19090")

steps = []
for line in open(log, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("event") == "step":
        steps.append(r)
if len(steps) <= skip + 1:
    sys.exit(f"only {len(steps)} steps in {log}")
t0, t1 = steps[skip - 1]["ts"] if skip else steps[0]["ts"] - steps[0]["seconds"], steps[-1]["ts"]
win = steps[skip:]
tokens = sum(s["tokens"] for s in win)
dur = t1 - t0

SERIES = {
    "rngd": f'furiosa_npu_hw_power{{label="rms"}}',
    "a100": f'DCGM_FI_DEV_POWER_USAGE{{Hostname="{node}"}}',
    "node": f'ipmi_dcmi_power_consumption_watts{{node="{node}"}}',
}


def raw(expr, start, end):
    q = urllib.parse.urlencode({"query": f"{expr}[{int(end - start) + 1}s]", "time": f"{end:.3f}"})
    with urllib.request.urlopen(f"{prom}/api/v1/query?{q}", timeout=60) as f:
        d = json.load(f)
    res = []
    for r in d["data"]["result"]:
        m = r["metric"]
        vals = [(float(t), float(v)) for t, v in r["values"] if start <= float(t) <= end]
        res.append((m, vals))
    return res


def ident(kind, m):
    if kind == "rngd":
        return f'{m.get("instance", "?")}/{m.get("device", "?")}'
    if kind == "a100":
        return f'gpu{m.get("gpu", "?")}'
    return "node"


def summarize(start, end, rows=None):
    out = {}
    for kind, expr in SERIES.items():
        for m, vals in raw(expr, start, end):
            if not vals:
                continue
            k = f"{kind}:{ident(kind, m)}"
            out[k] = {"mean_w": round(statistics.fmean(v for _, v in vals), 1), "n": len(vals),
                      "min_w": min(v for _, v in vals), "max_w": max(v for _, v in vals)}
            if rows is not None:
                rows += [(t, k, v) for t, v in vals]
    return out


rows = []
run = summarize(t0, t1, rows)
res = {"log": log, "node": node, "window": {"start": t0, "end": t1, "seconds": round(dur, 1),
                                            "steps": len(win), "first_step": win[0]["step"]},
       "tokens": tokens, "tok_s": round(tokens / dur, 2), "run": run}
if idle_before or idle_at:
    end = float(idle_at) if idle_at else steps[0]["ts"] - steps[0]["seconds"] - 600
    res["idle"] = summarize(end - (idle_before or 300), end)
    res["idle_window"] = {"start": end - (idle_before or 300), "end": end}
with open(out + ".power.json", "w") as f:
    json.dump(res, f, indent=1)
with open(out + ".power.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ts", "series", "watts"])
    w.writerows(sorted(rows))
print(json.dumps(res, indent=1))
