<div align="center">

<h1>npu-lora</h1>

<h3>추론 전용 NPU에서 LoRA와 전체 파인튜닝</h3>

<p>
  <a href="README.md">English</a> · <strong>한국어</strong>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L3%20execute%20%26%20serve-34d399" alt="ai-rm 계층: L3 실행·서빙"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10 이상"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.4 이상"></a>
  <img src="https://img.shields.io/badge/hardware-FuriosaAI%20RNGD-f97316" alt="하드웨어: FuriosaAI RNGD">
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-prototype-eab308" alt="성숙도: prototype"></a>
</p>

<p>
  <a href="#-주요-기능"><b>주요 기능</b></a> ·
  <a href="#-결과"><b>결과</b></a> ·
  <a href="#-빠른-시작"><b>빠른 시작</b></a> ·
  <a href="#-ai-rm-연동"><b>ai-rm 연동</b></a> ·
  <a href="#-동작-원리"><b>동작 원리</b></a> ·
  <a href="#-문서"><b>문서</b></a>
</p>

</div>

---

**npu-lora**는 FuriosaAI RNGD에서 Llama 계열 모델을 학습한다. RNGD의 SDK는 추론용 컴파일러와 런타임만
제공한다. npu-lora는 역전파를 일반 컴파일 그래프로 바꾸고, optimizer는 host에서 돌리며, vLLM과 llm-d가 그대로
서빙하는 표준 Hugging Face 산출물(PEFT 어댑터 또는 전체 모델)을 만든다. 같은 코드가 비교 실행용으로 CUDA와
CPU에서도 돈다.

[**ai-rm**](https://github.com/ai-computing/ai-rm)에서 이 저장소는 L3 학습 워크로드다. `npu-lora` workload
runner(`airm.workloads:npu-lora`)를 등록하고, 학습 처리량을 `airm_tok_s`로 내며, 학습 Job용 Helm chart를 제공한다.
KPI 측면에서는 이종자원 활용, 즉 추론 전용 NPU를 학습 자원으로 쓰는 데 기여한다.

## ✨ 주요 기능

<table>
<tr>
<td width="33%" valign="top">
<b>추론 컴파일러로 학습</b><br>
가중치를 그래프 입력으로 두고 역전파를 trace한 그래프로 만들어, RNGD 추론 컴파일러가 학습을 순전파 그래프처럼 실행한다.
</td>
<td width="33%" valign="top">
<b>A100과 같은 품질</b><br>
RNGD에서 학습한 Llama-3.1-8B LoRA가 A100의 같은 실행과 같은 결과를 낸다(NSMC 87.5% 대 87.6%, MMLU 둘 다 63.0%). 기울기는 CPU 기준과 일치한다.
</td>
<td width="33%" valign="top">
<b>표준 산출물, 클러스터 연동</b><br>
vLLM <code>--enable-lora</code>와 llm-d가 바로 읽는 PEFT 어댑터나 HF 전체 모델. 쿠버네티스 Job, Helm chart, Prometheus 지표.
</td>
</tr>
</table>

| | |
| --- | --- |
| **하드웨어** | FuriosaAI RNGD (furiosa-torch 2026.2) · NVIDIA GPU (비교용) · CPU (비교용) |
| **모델** | Llama 2/3/3.1/3.2 구조 (GQA, llama3 scaling 포함 RoPE, SwiGLU) |
| **모드** | LoRA (q/k/v/o) · CPU 오프로드를 쓰는 디코더 레이어 전체 파인튜닝 |
| **산출물** | PEFT 어댑터 (`adapter_model.safetensors`) · HF 전체 모델 · 병합 모델 |
| **연동** | vLLM 런타임 LoRA 로딩 · llm-d · 쿠버네티스 Job · [lmd-top](https://github.com/ai-computing/lmd-top) `--plan train` · Prometheus 지표 |

## 📊 결과

RNGD와 A100에서 같은 데이터·시드·하이퍼파라미터(Llama-3.1-8B-Instruct, q·v에 LoRA r=16, 400 step,
학습에 쓰지 않은 데이터로 vLLM 평가):

| | NSMC 시험 정답률 (n=1000) | MMLU 정답률 (n=400) |
| --- | ---: | ---: |
| 기본 모델 | 79.6% | 60.8% |
| **RNGD 학습** | **87.5%** | **63.0%** |
| A100 학습 | 87.6% | 63.0% |

- 기울기가 CPU 기준과 일치한다: cosine ≥ 0.9997 (Llama-3.1-8B 레이어 + 128k 어휘 head).
- RNGD 1장으로 Llama-3.1-8B LoRA를 128토큰 step당 약 0.7 s에 학습한다.
- 전체 파인튜닝은 Llama-3.2-1B로 검증했다.

원자료는 [docs/validation.md](docs/validation.md)와 [`experiments/`](experiments)에 있다.

> [!NOTE]
> **프로토타입.** LoRA 품질은 Llama-3.1-8B로, 전체 파인튜닝은 Llama-3.2-1B로 검증했다. Llama-3.1-8B 전체
> 파인튜닝은 카드 1장과 CPU 오프로드로 돌지만([step 시간 측정](experiments/2026-10-16-full8b-offload-step.md))
> 품질 비교는 아직 하지 않았다. 현재 상태와 열린 질문: [STATUS.md](STATUS.md).

## 🚀 빠른 시작

```bash
git clone https://github.com/ai-computing/npu-lora.git
cd npu-lora
pip install -e .
```

요구 사항:

- Python ≥ 3.10, PyTorch ≥ 2.4, `transformers`, `safetensors`, `huggingface_hub`.
- **RNGD:** `furiosa-torch`를 포함한 Furiosa SDK(2026.2.0에서 시험. `furiosaai/furiosa-llm` 이미지에 맞는
  조합이 들어 있다)와 컴파일러가 호출하는 aarch64 크로스 툴체인:

  ```bash
  apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross libc6-dev
  ```

- **평가:** `vllm` (`npu_lora.bench`에만 필요).

```bash
# 1. RNGD 기울기가 CPU 기준과 일치하는지 확인 (앞 2 레이어).
npu-lora check --model meta-llama/Llama-3.1-8B-Instruct --backend furiosa:0 --layers 2

# 2. LoRA 어댑터 학습. 데이터는 JSONL: {"text"} | {"prompt","response"} | {"messages"}.
npu-lora train --model meta-llama/Llama-3.1-8B-Instruct --data train.jsonl \
    --backend furiosa:0 --seq-len 128 --steps 400 --accum 4 --lr 2e-4 --out ./adapter

# 3. 실행 중인 vLLM(--enable-lora, VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)에 올리고 평가.
npu-lora serve --endpoint http://vllm:8000 --name my-adapter --path /models/adapter
npu-lora eval  --endpoint http://vllm:8000 --model my-adapter --data eval.jsonl
```

첫 실행은 학습 그래프를 조각 단위로 컴파일하고(Llama-8B: 64코어 호스트에서 약 1.5시간) `~/.cache/npu-lora`에
캐시한다. 이후 실행은 바로 시작한다. `npu-lora compile`은 장치를 열지 않고 캐시를 채우므로 아무 호스트에서나
컴파일한 뒤 캐시를 복사할 수 있다.

<details>
<summary><b>전체 파인튜닝</b></summary>
<br>

```bash
npu-lora train --full --model unsloth/Llama-3.2-1B-Instruct --data mix.jsonl \
    --backend furiosa:0 --seq-len 128 --steps 800 --accum 4 --lr 1e-5 --out ./model
```

디코더 레이어 가중치를 학습하고 임베딩, `lm_head`, 마지막 norm은 고정한다. 결과는 완전한 Hugging Face 모델
디렉터리다. 좁은 과제에는 범용 지시문 데이터를 섞는다. 한 과제로만 학습하면 장치와 상관없이 범용 능력이
무너진다([docs/validation.md](docs/validation.md) 참조).
</details>

<details>
<summary><b>쿠버네티스에서</b></summary>
<br>

```bash
npu-lora job --model meta-llama/Llama-3.1-8B-Instruct --data /mnt/store/datasets/train.jsonl \
    --name my-run --node <node> | kubectl apply -f -
```

Job은 공식 `furiosa-llm` 이미지에 ConfigMap으로 패키지를 마운트해 쓰고, `furiosa.ai/rngd` 카드 1장을 요청하며,
결과를 공유 모델 스토어에 쓰고, ServiceMonitor로 Prometheus 지표를 노출한다.
[lmd-top](https://github.com/ai-computing/lmd-top)은 `lmd-top --plan train --vendor furiosa --model … --set data=… [--set full=true]`로
같은 Job을 만든다.
</details>

<details>
<summary><b>Python API</b></summary>
<br>

```python
from npu_lora import LoRATrainer, LoRAConfig, TrainConfig
from npu_lora import backends, export
from npu_lora.data import PackedDataset
from transformers import AutoTokenizer

trainer = LoRATrainer(
    "meta-llama/Llama-3.1-8B-Instruct",
    backends.make("furiosa:0"),            # 또는 "cuda:0", "cpu"
    LoRAConfig(r=16, alpha=32, targets=("q", "v")),
    TrainConfig(seq_len=128, lr=2e-4, steps=400, accum=4),
)
trainer.setup()                             # 가중치 로드, 그래프 컴파일(또는 로드)

tok = AutoTokenizer.from_pretrained(trainer.path)
data = PackedDataset("train.jsonl", tok, seq_len=128)
for batch in data.batches(accum=4, steps=400):
    print(trainer.train_step(batch))        # {"step", "loss", "grad_norm", "tokens_per_s", ...}

export.save_peft(trainer, "./adapter")      # TrainConfig(full=True)이면 export.save_full(...)
```
</details>

<details>
<summary><b>명령과 옵션</b></summary>
<br>

| 명령 | 용도 |
| --- | --- |
| `npu-lora train` | LoRA 또는 `--full` 파인튜닝 |
| `npu-lora check` | 장치 기울기를 CPU 기준과 비교 |
| `npu-lora compile` | 장치 없이 학습 그래프를 컴파일해 캐시 |
| `npu-lora merge` | 어댑터를 기본 가중치에 병합 (NPU 서빙용, 재컴파일 필요) |
| `npu-lora serve` | 실행 중인 vLLM에 어댑터 로드 |
| `npu-lora eval` | OpenAI 호환 엔드포인트로 키워드 정답률 평가 |
| `npu-lora job` | 쿠버네티스 학습 Job 출력 |
| `python -m npu_lora.bench` | 기본 모델 / 어댑터 / 전체 모델의 vLLM 오프라인 벤치마크 |
| `python -m npu_lora.compare` | 학습 loss 곡선 두 개 비교 |
| `python -m npu_lora.hwprobe` | host↔device 대역폭과 전송/계산 겹침 측정 |

자주 쓰는 학습 옵션:

| 옵션 | 기본값 | 뜻 |
| --- | --- | --- |
| `--backend` | `furiosa:0` | `furiosa:N`, `furiosa:auto` (쿠버네티스가 배정한 카드), `cuda:N`, `cpu` |
| `--seq-len` | 512 | 시퀀스 길이. 컴파일 시점에 고정된다 (길이가 바뀌면 새로 컴파일) |
| `--rank`, `--alpha`, `--targets` | 16, 32, `q,v` | LoRA 설정 (`q,k,v,o`) |
| `--accum` | 1 | 기울기 누적 (micro-batch = 패킹된 시퀀스 1개) |
| `--full` | 꺼짐 | 디코더 레이어 가중치 전체 파인튜닝 |
| `--max-mm` | RNGD에서 2 | 컴파일 그래프 조각당 행렬곱 수 |
| `--metrics-port` | — | Prometheus 지표 서빙 (`airm_tok_s`, `npu_lora_loss`, `npu_lora_step_seconds`, …) |

학습 진행은 JSON 줄로 stdout과 `--log-file`에도 기록된다.
</details>

## 🔌 ai-rm 연동

npu-lora는 ai-rm 구성요소다(계층 `l3-serving`, kind `workload`, 성숙도 `prototype`). 설정은
[`ai-rm.yaml`](ai-rm.yaml). 추론 전용 AOT 컴파일 NPU인 FuriosaAI RNGD에서 Llama 계열 모델을 LoRA 또는 전체
파인튜닝하는 학습 워크로드이며, 결과는 HF PEFT 어댑터나 HF 전체 모델이다.

| Entry-point 그룹 | 이름 | 객체 |
| --- | --- | --- |
| `airm.workloads` | `npu-lora` | `npu_lora.airm:train` |

```python
from rm_core import Workload, registry
run = registry.load("airm.workloads", "npu-lora", backend="rngd")    # 생성자 kwargs = 설정 기본값
res = run(Workload(kind="train", model="meta-llama/Llama-3.1-8B-Instruct"),
          data="/mnt/store/datasets/nsmc.jsonl", out="/mnt/store/adapters/nsmc", dataset="nsmc", steps=400)
res["measurements"]      # [Measurement("tok_s", …, tags={"workload": "train", "dataset": "nsmc", …})]
```

runner는 기존 `npu-lora train`을 감쌀 뿐 학습 경로는 같다.

**설치·테스트**: `python -m pip install -e .`, `python tests/test_cpu.py`(CPU 회귀 테스트, NPU 불필요, PyTorch
필요), `python tests/test_airm.py`(ai-rm runner, CPU), `python -m rm_core.conformance`. RNGD 학습에는 Furiosa
SDK(`furiosa-torch`)가 필요하다.

<details>
<summary><b>쿠버네티스, llm-d, Helm chart</b></summary>
<br>

학습은 쿠버네티스 Job으로 돈다. `npu-lora job … | kubectl apply -f -`가 ConfigMap(패키지), Job(`furiosa.ai/rngd`
1장, `furiosaai/furiosa-llm` 이미지, privileged), Service, ServiceMonitor를 만들고 결과를 공유 모델 스토어
PVC(`model-store`)에 쓴다. 산출 어댑터는 vLLM `--enable-lora`와 llm-d LoRA 라우팅이 그대로 읽고, `npu-lora serve`로
실행 중인 vLLM에 재시작 없이 올린다. lmd-top은 `lmd-top --plan train`으로 같은 Job을 만든다.

ai-rm umbrella chart는 Helm chart [`deploy/helm/npu-lora`](deploy/helm/npu-lora)를 쓴다. 이 chart는
`train.backend`(`rngd|cuda|cpu`)에 따라 노드를 `airm.ai-computing.io/<vendor>.<model>` 레이블로 고르고
(rngd: `airm.ai-computing.io/furiosa.RNGD`, 그 외는 `nodeAffinity`로 지정), 데이터·체크포인트·캐시를 모델
스토어 PVC(`store.pvc`)에 둔다. 이미지에 npu-lora가 없으면 시작할 때 `install.pip`로 설치한다. 지표 포트에는
`airm.ai-computing.io/scrape: "true"` 어노테이션이 붙는다.

```bash
helm template deploy/helm/npu-lora --set train.model=… --set train.data=…
```
</details>

<details>
<summary><b>Workload runner 설정</b></summary>
<br>

`Workload`는 `kind="train"`, `precision="bf16"`만 받는다(다르면 `ContractError`).

| 키 | 뜻 (기본값) |
| --- | --- |
| `data`, `out` | 필수. 학습 JSONL, 출력 디렉터리 |
| `backend` | `rngd` · `rngd:N` · `cuda[:N]` · `cpu` (`rngd`) |
| `dataset` | 지표 태그 `dataset` 값 (없음) |
| `rank`, `alpha`, `targets`, `seq_len`, `steps`, `accum`, `lr`, `warmup`, `grad_clip`, `seed`, `full`, `save_every`, `merge_out`, `run_name`, `log_file`, `metrics_port`, `cache_dir`, `tactic`, `max_shapes`, `max_mm` | `npu-lora train`의 같은 이름 옵션 |
| `kv_transport` | 학습에는 KV 경로가 없다. 비워 둔다(주면 `ContractError`) |
</details>

<details>
<summary><b>지표</b></summary>
<br>

rm-core `METRICS` 이름을 쓴다. 학습 처리량 `tok_s`(Prometheus `airm_tok_s`, 태그 `component=npu-lora`,
`workload=train`, `model`, `vendor`, `run_id`; runner 반환값에도 들어 있다). 정확도 `accuracy_pct`는
`npu_lora.bench` 결과 JSON의 `measurements`(태그 `dataset=nsmc|mmlu`, `run_id`=변형 이름)에 있다. rm-core에 없는
양은 `npu_lora_*`로 낸다(`npu_lora_loss`, `npu_lora_grad_norm`, `npu_lora_step_seconds` 등).

**Deprecated**: 예전 처리량 시리즈 `npu_lora_tokens_per_second`(라벨 `run`)는 `airm_tok_s`와 같은 값으로 프로토콜
0.2 전까지만 함께 낸다(PROTOCOL §6 이행 규칙). 소비자는 `airm_tok_s`를 먼저 읽고 없을 때만 예전 이름을 쓴다.
step 시간의 공통 이름은 rm-core 후보 `step_ms`다. `power_w`는 아직 내지 않는다. 장치는 `furiosa.ai/rngd` 리소스와
`NPU_LORA_RNGD`/`NPU_LORA_RNGD_FILE`로 고른다.
</details>

<details>
<summary><b>사이트 설정 (<code>cluster.local.env</code>)</b></summary>
<br>

라이브러리와 CLI는 사이트 값을 쓰지 않는다(`--endpoint` 등은 인자로 받는다). 데모 데이터 생성기
`examples/cluster_facts.py`만 아래 값을 읽는다. [`cluster.example.env`](cluster.example.env)를 `cluster.local.env`로
복사해 채운다.

| 변수 | 뜻 |
| --- | --- |
| `MODEL_STORE_NAS_HOST` | 모델 스토어 NAS 주소 |
| `LLMD_GATEWAY_HOST` | llm-d 게이트웨이 주소 |
| `PROMETHEUS_HOST`, `PROMETHEUS_PORT` | 클러스터 Prometheus 주소와 포트 |
</details>

## 🏗 동작 원리

1. **가중치가 그래프 입력이다.** 그래서 컴파일한 디코더 레이어 그래프 하나가 모든 레이어에 쓰인다.
2. **역전파는 trace한 그래프다**(`torch.func.vjp` 위의 `make_fx`). 컴파일러가 처리하지 못하는 패턴을 의미를
   보존하는 재작성으로 바꾼 뒤, 추론 컴파일러가 순전파 그래프처럼 실행한다.
3. **그래프는 행렬곱 2개 단위 조각으로 컴파일한다.** 큰 융합 그래프는 컴파일러 자원을 소진한다.
4. **optimizer는 host에서 돈다.** 전체 파인튜닝은 device→host 기울기 전송, host AdamW, host→device 업로드를
   역전파와 다음 순전파에 겹친다.

자세한 내용: [docs/design.md](docs/design.md). 컴파일러 문제와 우회:
[docs/compiler-workarounds.md](docs/compiler-workarounds.md).

<details>
<summary><b>개발</b></summary>
<br>

```bash
python tests/test_cpu.py        # CPU 회귀 테스트 (NPU 불필요)
```

테스트는 개발 중 조용히 틀린 결과를 냈던 문제를 고정한다. 그래프 재작성과 분할이 값을 바꾸지 않아야 하고,
중복 샘플 텐서가 입력을 합치지 않아야 하며, 어휘 청크 loss와 RoPE scaling이 Hugging Face와 일치해야 하고,
전체 파인튜닝 파이프라인이 동기 갱신과 같아야 한다.

새 모델이나 SDK 버전을 올릴 때:

1. `npu-lora check`가 통과해야 한다 (장치 대 CPU 기울기).
2. 컴파일이 실패하면 `npu_lora.diagnose.first_failing_node`로 처음 실패하는 연산을 찾고, 의미를 보존하는
   재작성을 `graph.PASSES`에 추가한다.
3. 컴파일은 되는데 결과가 틀리면 `npu_lora.diagnose.compare_parts`로 컴파일된 조각마다 같은 입력에서 CPU와 비교한다.

`scripts/vendor-into-lmd-top.sh`는 패키지를 lmd-top에 복사한다. lmd-top은 이를 자신이 만드는 Job에 넣는다.
</details>

### 한계

- Llama 구조만 지원한다. 다른 구조는 레이어를 순수 함수로 작성해야 한다(`npu_lora/llama.py` 참조).
- micro-batch당 패킹된 시퀀스 1개다. 더 큰 배치는 `--accum`을 쓴다.
- 첫 컴파일이 느리고(8B 모델은 수 시간) 시퀀스 길이마다 따로 컴파일한다.
- furiosa-torch는 privileged 컨테이너에서만 장치를 연다.
- Rebellions ATOM은 아직 지원하지 않는다. 같은 방식(`rebel.compile_from_torch`)이 적용된다.

## 📚 문서

| 문서 | 내용 |
| --- | --- |
| [docs/design.md](docs/design.md) | 추론 전용 NPU에서 학습이 되는 이유, 어휘 청크 loss, CPU 오프로드 파이프라인, Job 구성 (영문) |
| [docs/validation.md](docs/validation.md) | PoC, RNGD 대 A100 품질, 전체 파인튜닝 결과 |
| [docs/compiler-workarounds.md](docs/compiler-workarounds.md) | furiosa-torch 2026.2 컴파일러 문제와 우회 |
| [`experiments/`](experiments) | 실험 기록과 원자료 (`experiments/data/`) |
| [STATUS.md](STATUS.md) | 현재 상태, 발견 사항, 열린 질문 |

ETRI llm-d 클러스터에서 나온 연구 코드다.
