"""Build W7 experiment Jobs from the npu-train checkout (npu_lora.k8s) with experiment-only changes.

usage: gen.py NAME KIND NODE [--key value ...] -- <npu_lora.cli args...>
  KIND: rngd (1 RNGD card, privileged, init container records the card) | rngd2 (2 cards)
        cuda (1 A100, vllm-openai image) | cpu (no accelerator; compile-only)
Writes YAML to stdout. All objects carry label w7-exp=true for cleanup.
"""
import json
import sys

sys.path.insert(0, sys.argv.pop(1))          # npu-train checkout
from npu_lora import k8s                       # noqa: E402

TOL = [{"key": "airm.ai-computing.io/experiment", "operator": "Equal", "value": "true", "effect": "NoSchedule"}]
CUDA_IMAGE = "vllm/vllm-openai:v0.11.0"

name, kind, node = sys.argv[1:4]
rest = sys.argv[4:]
opts, cmd = {}, []
if "--" in rest:
    i = rest.index("--")
    cmd = rest[i + 1:]
    rest = rest[:i]
for k, v in zip(rest[::2], rest[1::2]):
    opts[k.lstrip("-")] = v

ns = "llm-serving"
app = f"w7-{name}"
labels = {"app": app, "w7-exp": "true"}
cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": f"{app}-src", "namespace": ns, "labels": labels},
      "data": k8s.package_files()}
if opts.get("extra"):                          # extra script(s) next to the package: /opt/npu-lora/npu_lora/<name>
    import os
    for f in opts["extra"].split(","):
        cm["data"][os.path.basename(f)] = open(f, encoding="utf-8").read()
volumes = [{"name": "src", "configMap": {"name": f"{app}-src"}},
           {"name": "store", "persistentVolumeClaim": {"claimName": "model-store"}},
           {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
           {"name": "rngd", "emptyDir": {}}]
mounts = [{"name": "src", "mountPath": "/opt/npu-lora/npu_lora", "readOnly": True},
          {"name": "store", "mountPath": "/mnt/store"},
          {"name": "shm", "mountPath": "/dev/shm"},
          {"name": "rngd", "mountPath": "/run/npu-lora"}]
cache = opts.get("cache", "pvc")
if cache == "pvc":
    mounts.append({"name": "store", "mountPath": "/cache", "subPath": "npu-lora-cache"})
else:                                          # hostPath on the node
    volumes.append({"name": "cache", "hostPath": {"path": cache, "type": "DirectoryOrCreate"}})
    mounts.append({"name": "cache", "mountPath": "/cache"})
env = [{"name": "PYTHONPATH", "value": "/opt/npu-lora"},
       {"name": "NPU_LORA_RNGD_FILE", "value": "/run/npu-lora/rngd"},
       {"name": "PYTHONUNBUFFERED", "value": "1"},
       {"name": "HF_HUB_OFFLINE", "value": "1"},
       {"name": "HF_HOME", "value": opts.get("hf", "/mnt/store/hub")},
       {"name": "W7_RUN", "value": name}]
for kv in opts.get("env", "").split(","):
    if kv:
        k, v = kv.split("=", 1)
        env.append({"name": k, "value": v})
log = opts.get("log", f"/mnt/store/w7/logs/{name}.out")
shell = (f'mkdir -p /mnt/store/w7/logs && echo "start $(date +%s) $(hostname)" > {log}.meta && '
         f'(python3 - <<"P" >> {log}.meta 2>&1\n'
         'import importlib.metadata as m\n'
         'for p in ("torch","furiosa-torch","transformers","vllm","furiosa-llm"):\n'
         '    try: print(p, m.version(p))\n'
         '    except Exception: pass\n'
         'P\n) ; ')
if kind != "cuda":
    shell += k8s.TOOLCHAIN + " && "
shell += 'set -o pipefail; "$@" 2>&1 | tee ' + log + '; rc=$?; echo "end $(date +%s) rc=$rc" >> ' + log + '.meta; exit $rc'
ncards = 2 if kind == "rngd2" else 1
res = {"requests": {"cpu": opts.get("cpu", "16"), "memory": opts.get("mem", "64Gi")}, "limits": {}}
c = {"name": "run", "imagePullPolicy": "IfNotPresent", "command": ["bash", "-c", shell, "--"] + cmd,
     "env": env, "volumeMounts": mounts, "resources": res}
pod = {"restartPolicy": "Never", "tolerations": TOL, "containers": [c], "volumes": volumes,
       "nodeSelector": {"kubernetes.io/hostname": node}}
if kind.startswith("rngd"):
    c["image"] = opts.get("image", k8s.IMAGE)
    c["securityContext"] = {"privileged": True}
    res["limits"]["furiosa.ai/rngd"] = ncards
    detect = (f"ls /dev/rngd | grep -oE '^npu[0-9]+' | sort -u | head -{ncards} | paste -sd, > /run/npu-lora/rngd"
              " && echo assigned $(cat /run/npu-lora/rngd)")
    pod["initContainers"] = [{"name": "assigned-rngd", "image": "busybox:1.36", "imagePullPolicy": "IfNotPresent",
                              "command": ["sh", "-c", detect], "resources": {"limits": {"furiosa.ai/rngd": ncards}},
                              "volumeMounts": [{"name": "rngd", "mountPath": "/run/npu-lora"}]}]
elif kind == "cuda":
    c["image"] = opts.get("image", CUDA_IMAGE)
    res["limits"]["nvidia.com/gpu"] = 1
    pod["runtimeClassName"] = "nvidia"
else:
    c["image"] = opts.get("image", k8s.IMAGE)
if opts.get("memlimit"):
    res["limits"]["memory"] = opts["memlimit"]
job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": app, "namespace": ns, "labels": labels},
       "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": labels}, "spec": pod}}}
print(json.dumps({"apiVersion": "v1", "kind": "List", "items": [cm, job]}, ensure_ascii=False))
