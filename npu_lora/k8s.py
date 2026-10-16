"""쿠버네티스 학습 Job 매니페스트.

공식 furiosa-llm 이미지에 학습에 필요한 패키지(furiosa-torch, torch, transformers)가 이미 있으므로,
이미지를 새로 만들지 않고 npu-lora 소스를 ConfigMap 으로 얹는다. 매니페스트는 문자열을 이어
붙이지 않고 dict 를 직렬화한다 — 모델 id·경로 같은 값이 YAML 문법에 끼어들 수 없다.

  npu-lora job --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/x.jsonl \
      --name demo | kubectl apply -f -
"""
import json
import os
import re

IMAGE = "furiosaai/furiosa-llm:latest"
STORE_PVC = "model-store"
STORE = "/mnt/store"
METRICS_PORT = 9400
TOOLCHAIN = ("command -v aarch64-linux-gnu-gcc >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
             "apt-get install -y -qq --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross libc6-dev)")


def _name(s):
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return s[:52] or "run"


def package_files():
    """npu_lora 패키지의 .py 파일 {이름: 내용} — ConfigMap 키가 된다(하위 디렉터리 없음)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: open(os.path.join(here, f)).read() for f in sorted(os.listdir(here)) if f.endswith(".py")}


def build(model, data, name, ns="llm-serving", steps=100, seq_len=512, rank=16, alpha=32.0,
          targets="q,v", lr=2e-4, accum=1, image=IMAGE, node=None, hf_host_path=None,
          cache_host_path=None, out_dir=None, merge=False, extra_args=()):
    """(ConfigMap, Job, Service, ServiceMonitor) dict 리스트.

    - 데이터·출력·컴파일 캐시는 공유 모델 스토어 PVC 아래에 둔다(어느 노드에서 돌아도 같다).
    - hf_host_path 를 주면 HF 캐시를 그 노드의 hostPath 에서 읽는다(스토어에 가중치가 없을 때).
    """
    run = _name(name)
    app = f"npu-lora-{run}"
    repo_dir = model.replace("/", "--")
    out_dir = out_dir or f"{STORE}/adapters/{repo_dir}/{run}"
    labels = {"app": app, "app.kubernetes.io/part-of": "npu-lora", "npu-lora/run": run}
    cmd = ["python3", "-m", "npu_lora.cli", "train",
           "--model", model, "--data", data, "--out", out_dir, "--backend", "furiosa:auto",
           "--steps", str(steps), "--seq-len", str(seq_len), "--rank", str(rank), "--alpha", str(alpha),
           "--targets", targets, "--lr", str(lr), "--accum", str(accum),
           "--run-name", run, "--metrics-port", str(METRICS_PORT),
           "--log-file", f"{out_dir}/train.jsonl",
           "--cache-dir", "/cache/furiosa"]
    if merge:
        cmd += ["--merge-out", f"{out_dir}/merged"]
    cmd += list(extra_args)

    volumes = [{"name": "rngd", "emptyDir": {}},
               {"name": "src", "configMap": {"name": f"{app}-src"}},
               {"name": "store", "persistentVolumeClaim": {"claimName": STORE_PVC}}]
    mounts = [{"name": "rngd", "mountPath": "/run/npu-lora"},
              {"name": "src", "mountPath": "/opt/npu-lora/npu_lora", "readOnly": True},
              {"name": "store", "mountPath": STORE}]
    env = [{"name": "PYTHONPATH", "value": "/opt/npu-lora"},
           {"name": "NPU_LORA_RNGD_FILE", "value": "/run/npu-lora/rngd"},
           {"name": "PYTHONUNBUFFERED", "value": "1"},
           {"name": "HF_HUB_OFFLINE", "value": "1"}]
    if hf_host_path:
        volumes.append({"name": "hf", "hostPath": {"path": hf_host_path, "type": "Directory"}})
        mounts.append({"name": "hf", "mountPath": "/hf", "readOnly": True})
        env.append({"name": "HF_HOME", "value": "/hf"})
    else:
        env.append({"name": "HF_HOME", "value": f"{STORE}/hub"})
    if cache_host_path:
        volumes.append({"name": "cache", "hostPath": {"path": cache_host_path, "type": "DirectoryOrCreate"}})
    else:
        volumes.append({"name": "cache", "persistentVolumeClaim": {"claimName": STORE_PVC}})
    mounts.append({"name": "cache", "mountPath": "/cache",
                   **({} if cache_host_path else {"subPath": "npu-lora-cache"})})

    # furiosa-torch 는 privileged 에서만 장치를 연다. privileged 컨테이너는 노드의 RNGD 를 전부 보므로,
    # privileged 가 아닌 init 컨테이너가 device plugin 이 배정한 카드 이름을 적고 학습은 그 카드만 쓴다.
    detect = ("ls /dev/rngd | grep -oE '^npu[0-9]+' | sort -u | head -1 > /run/npu-lora/rngd"
              " && echo assigned $(cat /run/npu-lora/rngd)")
    pod = {
        "restartPolicy": "Never",
        "initContainers": [{
            "name": "assigned-rngd", "image": "busybox:1.36", "imagePullPolicy": "IfNotPresent",
            "command": ["sh", "-c", detect],
            "resources": {"limits": {"furiosa.ai/rngd": 1}},
            "volumeMounts": [{"name": "rngd", "mountPath": "/run/npu-lora"}],
        }],
        "containers": [{
            "name": "train", "image": image, "imagePullPolicy": "IfNotPresent",
            # 공식 furiosa-llm 이미지에는 컴파일러가 호출하는 aarch64 크로스 툴체인이 없다(서빙은 미리 빌드한
            # artifact 만 쓴다). 없으면 첫 컴파일이 "No such file or directory" 로 실패한다.
            "command": ["bash", "-c", TOOLCHAIN + ' && mkdir -p "$OUT" && exec "$@"', "--"] + cmd,
            "env": env + [{"name": "OUT", "value": out_dir}],
            "ports": [{"name": "metrics", "containerPort": METRICS_PORT}],
            "securityContext": {"privileged": True},
            "resources": {"limits": {"furiosa.ai/rngd": 1},
                          "requests": {"cpu": "8", "memory": "64Gi"}},
            "volumeMounts": mounts,
        }],
        "volumes": volumes,
    }
    if node:
        pod["nodeSelector"] = {"kubernetes.io/hostname": node}

    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": f"{app}-src", "namespace": ns, "labels": labels},
          "data": package_files()}
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": app, "namespace": ns, "labels": labels,
                        "annotations": {"npu-lora/model": model, "npu-lora/data": data, "npu-lora/out": out_dir}},
           "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 86400,
                    "template": {"metadata": {"labels": labels}, "spec": pod}}}
    svc = {"apiVersion": "v1", "kind": "Service",
           "metadata": {"name": app, "namespace": ns, "labels": labels},
           "spec": {"selector": {"app": app}, "ports": [{"name": "metrics", "port": METRICS_PORT}]}}
    sm = {"apiVersion": "monitoring.coreos.com/v1", "kind": "ServiceMonitor",
          # Job 과 같은 네임스페이스: 이 클러스터의 Prometheus 는 모든 네임스페이스의 ServiceMonitor 를 고른다.
          "metadata": {"name": app, "namespace": ns,
                       "labels": {"release": "kube-prometheus-stack", **labels}},
          "spec": {"namespaceSelector": {"matchNames": [ns]}, "selector": {"matchLabels": {"app": app}},
                   "endpoints": [{"port": "metrics", "path": "/metrics", "interval": "15s"}]}}
    return [cm, job, svc, sm]


def to_yaml(docs):
    """PyYAML 이 없어도 되도록 JSON 으로 쓴다 — JSON 은 YAML 의 부분집합이라 kubectl 이 그대로 읽는다."""
    try:
        import yaml
        return "---\n".join(yaml.safe_dump(d, sort_keys=False, allow_unicode=True) for d in docs)
    except ImportError:
        return "\n---\n".join(json.dumps(d, ensure_ascii=False, indent=1) for d in docs)
