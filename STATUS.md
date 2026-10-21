# STATUS — RNGD NPU 파인튜닝 학습

<!-- research-hub:header:begin (maintained by .research/report.py; keep the "- **Key**: value" format) -->
- **State**: active
- **Updated**: 2026-10-21 21:57+09:00
- **Agent**: claude@DESKTOP-R49IQ68
- **Task**: #35 MoE hot experts batched
- **Last result**: batched hot experts 13.0 s/step vs host-only 13.5 (alternating steps, Qwen3-30B-A3B, RNGD); per-expert main 42.4 vs 17.2
- **Next**: more rows per call (longer S), hot fraction 0.5, drop the recomputed forward
- **Blockers**: none
<!-- research-hub:header:end -->

## Goal

추론 전용 AOT 컴파일 NPU인 Furiosa RNGD에서 LoRA와 full fine-tuning을 실용 수준으로 돌리고, 같은 조건의 A100 학습과 품질·속도를 비교한다. 구현은 공개 저장소 [ai-computing/npu-lora](https://github.com/ai-computing/npu-lora)의 `npu_lora` 패키지(백엔드 furiosa/cuda/cpu)다. 결과를 llm-d 서빙(vLLM `--enable-lora`)과 lmd-top(`--plan train`)에 붙이고, ai-rm 구성요소(`ai-rm.yaml`, protocol 0.1)로 내는 것까지 범위에 넣는다 (`README.md`).

## Findings

최신순. 출처를 괄호에 적었다. 레포에 원자료가 없는 수치는 그렇다고 밝혔다.

- **2026-10-19 · ai-rm protocol v0.1 통합이 main에 머지됐다**(PR #1, merge `f9b7bc5`; ai-rm이 이 커밋을 고정). workload runner `npu_lora.airm:train`, 처리량 지표 `airm_tok_s`, Helm 차트 `deploy/helm/npu-lora`가 들어갔다. 예전 시리즈 `npu_lora_tokens_per_second`는 deprecated이고 프로토콜 0.2 전까지만 함께 낸다 (`README.md` 「ai-rm」, `npu_lora/airm.py`, `tests/test_airm.py`).
- **2026-10-16 · RNGD LoRA 학습 품질이 A100 학습과 같다.** Llama-3.1-8B, S=128, r=16 q·v, lr 2e-4, 400 step × accum 4, 같은 데이터와 시드로 비교했다. NSMC 시험(n=1000)은 기본 79.6%, RNGD 87.5%, A100 87.6%였다. MMLU(n=400)는 기본 60.8%, RNGD와 A100 모두 63.0%였다. loss 곡선 40 step 구간 평균의 상대차 중앙값은 2.8%, 마지막 구간은 0.0676 대 0.0667이었다. 평가는 `npu_lora.bench`(vLLM 오프라인 엔진, temperature 0), 곡선 비교는 `npu_lora.compare`로 했다 (`experiments/2026-10-16-nsmc-lora-rngd-vs-a100.md`, `docs/validation.md`).
- **2026-10-16 · 품질 검증 중 버그 2건을 고쳤다.** llama3 `rope_scaling`이 적용되지 않았다(HF 대비 오차 1.8e-4 → 3.5e-7). 긴 예제는 응답째 잘렸다(이제 프롬프트 쪽을 덜어낸다) (`docs/validation.md`).
- **full fine-tuning 모드가 있고 1B로 품질을 실증했다.** `--full`(lmd-top `--set full=true`)은 레이어 가중치 9종을 학습하고 임베딩·lm_head는 고정한다. 결과는 HF 전체 모델로 저장한다 (`npu_lora/cli.py`, `trainer.py`, `export.py` `save_full`). 품질 비교는 `unsloth/Llama-3.2-1B-Instruct`로 했다(8B는 아래 오프로드 항목). 1B 기본 성능은 NSMC 62.8%, MMLU 43.25%였다. RNGD 기울기 점검(`npu-lora check --full`, 2층)은 최대 2.1%로 통과했고, 동기 갱신 step은 약 5 s였다 (`experiments/2026-10-16-full1b-nsmc-task-only.md`). 한 과제로만 full FT를 하면 모든 질문에 그 과제 형식으로 답하는 망각이 생겼다(1B MMLU 43% → RNGD 20.25%, A100 8.5%). 그래서 범용 지시문(Alpaca)을 1:1로 섞는 replay 데이터를 만들었다 (`examples/nsmc.py`). 같은 조건(1B, replay, 800 step)에서 RNGD는 NSMC 78.2%·MMLU 36.25%, A100은 79.0%·36.75%였고, loss 곡선 상대차는 1.2%였다. RNGD full FT는 GPU와 같은 품질을 낸다. 오프로드 파이프라인 적용 후 RNGD 1B step은 3.85 s다 (`experiments/2026-10-16-full1b-nsmc-replay.md`).
- **2026-10-15 · RNGD 1장으로 Llama-3.1-8B 32층 LoRA 학습 PoC를 마쳤다.** 0.7 s/step(약 180 tok/s)였고 200 step에 2분 36초, loss 3.29 → ~0이었다. 학습에 없는 질문은 0/10에서 10/10이 됐다(A100 vLLM 런타임 LoRA로 평가, CUDA 백엔드 학습도 10/10). RNGD 대 CPU 기울기는 cos ≥ 0.9997, 상대오차 ≤ 2.5%(Llama-3.1-8B 2층 + 128k 어휘 head, loss 1.9e-4)였다. 함수형 Llama 대 HF loss 상대오차는 1.7e-5였다 (`docs/validation.md` 「PoC」; 별도 experiments/ 기록은 없다).
- **컴파일 비용과 구조.** 첫 컴파일에 약 1.5시간이 걸리고, 이후에는 EDF 캐시(15 MB)로 바로 시작한다 (`docs/validation.md`). 그래프는 행렬곱 2개 단위 조각으로 컴파일해 32층과 모든 어휘 조각에 재사용한다 (`docs/design.md`). Llama-8B 레이어가 순전파 8조각, 역전파 19조각이라는 수치와 어댑터(r=16, q·v) 13.6 MB는 import 당시 README(`ef0b8ee`)에만 있고 현재 문서에서는 빠졌다.
- **furiosa-torch 2026.2 컴파일러 문제와 대응을 15건 정리했다.** 그중 3건은 "조용히 틀린 값"을 낸다: 조각 출력 전치, 컴파일 샘플 입력 stride 고정, make_fx 중복 입력 병합. 이 경우는 컴파일 오류로 드러나지 않아 `npu_lora.diagnose.compare_parts`로 조각마다 CPU와 비교해 잡는다 (`docs/compiler-workarounds.md`).
- **llm-d·lmd-top 연동.** 출력은 HF PEFT 형식이라 vLLM `--enable-lora`와 llm-d LoRA 라우팅이 그대로 읽는다. `npu-lora job`, Helm 차트, `lmd-top --plan train`으로 클러스터 Job을 만든다. `--metrics-port`로 Prometheus 지표(`airm_tok_s`, `npu_lora_loss`, `npu_lora_step_seconds` 등)를 내고, `npu-lora job`이 ServiceMonitor도 만든다 (`README.md` 「Commands」「ai-rm」). lmd-top은 아직 예전 `npu_lora_tokens_per_second`를 읽는다(NPUTRAIN-30). 테스트는 `tests/test_cpu.py` 11개, `tests/test_airm.py` 4개다.

## Open questions

- **8B full fine-tuning은 카드 1장 + CPU 오프로드로 돈다.** fp32 master·AdamW 상태(~100 GB)는 host, 카드에는 bf16 가중치와 누적 기울기(32 GB)만 둔다. 커밋된 기록은 S=512 accum 1 10.33 s/step(49.6 tok/s), accum 4 13.13 s/step(156 tok/s)이다 (`experiments/2026-10-16-full8b-offload-step.md`). 2026-10-19 후속 측정(같은 파일의 미커밋 절)에서는 상주 host 버퍼로 RNGD accum 1 128 tok/s, accum 4 315 tok/s, 같은 코드의 A100은 153 / 477 tok/s였고 loss는 두 장치가 일치했다. 처음 병목은 PCIe가 아니라 `t.cpu()`가 매번 host 메모리를 새로 잡는 비용이었다(3.0 → 19.9 GB/s). 남은 ~3 s 고정비는 host AdamW와 전송이 나눠 갖는다.
- **미커밋 작업이 남아 있다.** 8B full FT 오프로드의 상주 host 버퍼·CUDA 복사 stream(`npu_lora/backends.py` `host_buffer`/`to_host(out=)`, `npu_lora/trainer.py` 레이어별 버퍼), 그 설명(`docs/design.md`), A100 대조와 전송 개선 결과(`experiments/2026-10-16-full8b-offload-step.md` 후속 절, `experiments/data/{old8b-rngd,buf8b-rngd,full8b-cuda,full8b-cudapin,full8b-cudabuf}.jsonl`)가 main(`f9b7bc5`) 위에 커밋되지 않은 채 있다.
- **첫 컴파일에 1.5시간이 든다.** 우회 패스가 다른 모델과 SDK(예: Furiosa 2026.4)에서도 일반화되는지 확인하지 않았다 (`docs/validation.md`, `docs/compiler-workarounds.md`).
- **지원 범위가 좁다.** 지금은 Llama 계열(GQA, RoPE, SwiGLU)만 된다. micro-batch는 시퀀스 1개이고, 시퀀스 길이는 컴파일 시점에 고정된다 (`README.md` 「Limitations」).
- **RBLN(ATOM) 백엔드가 없다.** 같은 방식(`rebel.compile_from_torch`)이 가능하지만 이 클러스터의 RBLN 컴파일 경로가 고장 나 있어 넣지 않았다 (`README.md` 「Limitations」, llm-d-setup 레포 `lmd-top/docs/RBLN-COMPILE-INCIDENT.md`).
- **Job 구성에 제약이 있다.** furiosa-torch는 privileged 컨테이너에서만 장치를 연다. 그래서 init 컨테이너가 배정 카드를 적고 학습 컨테이너가 서수(`rngd:N`)를 쓴다 (`docs/design.md`).
- **새 모델마다 검증이 필요하다.** 조용히 틀리는 컴파일 결과는 오류로 드러나지 않으므로 `npu-lora check`와 `diagnose.compare_parts`를 매번 돌려야 한다.
- **실험 기록 범위.** `experiments/`에는 LoRA 품질 비교, 1B full FT 2건(과제만·replay), 8B 오프로드 step 측정의 기록 4건과 원자료(`experiments/data/`)가 있다. 2026-10-15 LoRA PoC 수치는 `docs/validation.md`에만 있고 별도 실험 기록은 없다.

## Log

<!-- research-hub:log (report.py inserts new entries below this line, newest first) -->

### 2026-10-21 21:57+09:00 — #35 MoE hot experts batched

- Result: batched hot experts 13.0 s/step vs host-only 13.5 (alternating steps, Qwen3-30B-A3B, RNGD); per-expert main 42.4 vs 17.2
- Next: more rows per call (longer S), hot fraction 0.5, drop the recomputed forward
- Metrics: step_time_s=13.0
- Experiments: experiments/2026-10-21-moe-batched-hot-experts.md
- By: claude@DESKTOP-R49IQ68 · perf/35-batch-hot-experts@9d5e498

### 2026-10-21 20:20+09:00 — #33 Qwen2 gradient check on RNGD

- Result: npu-lora check passes (worst 3.1%, was 44%); 0 negative-loss steps in 400 (was 11); loss curve matches A100
- Next: Llama-3.1-8B check with the new head; Qwen3
- Metrics: step_time_s=2.74
- Experiments: experiments/2026-10-21-qwen2-grad-precision.md
- By: claude@DESKTOP-R49IQ68 · fix/33-qwen-lora-grad@a7e206a

### 2026-10-21 15:37+09:00 — #34 resident device AdamW + host-throughput step model

- Result: compile --adamw-only 150 s (was >2.5 h); 2 RNGD x 8 resident layers 4.57 s/step vs 1-card offload 5.22 s (+14% tok/s, steps 3-7, intermittent OOM #40); --update-workers 8 -8%; host-throughput model within 15% on 6 configs
- Next: fix resident OOM (#40), then 200-step run
- Metrics: tok_s=448
- Params: seq_len=512, accum=4, resident_layers=8, cards=2, sdk=furiosa-torch-2026.2.0
- Experiments: experiments/2026-10-21-resident-adamw.md
- By: claude@control-node · perf/34-resident-adamw-compile@8539644-dirty

### 2026-10-19 09:53+09:00 — NPUTRAIN-32 ✅

- Result: PR #1 갱신(931a189, airm/v0.1): npu_lora_tokens_per_second를 airm_tok_s와 함께 프로토콜 0.2 전까지 재출력(README deprecated 표시), rm-core v0.1.1. 검증: conformance PASS 3/FAIL 0, test_airm 4/4, test_cpu 11/11, helm template OK, check_sensitive clean. airm.workloads conformance는 0.1.1에서도 KeyError(NPUTRAIN-31)
- Next: PR 머지 후 lmd-top을 airm_tok_s 우선 읽기로 전환(NPUTRAIN-30), 프로토콜 0.2에서 예전 시리즈 제거
- By: claude@control-node · main@5ba96b2-dirty

### 2026-10-19 09:45+09:00 — NPUTRAIN-19

- Result: 8B full FT S=512 accum4: RNGD 315 tok/s vs A100 477 tok/s(같은 오프로드 코드, loss 일치). 상주 host 버퍼로 RNGD 2.1배(t.cpu() D2H 3.0→19.9 GB/s), A100은 복사 stream+pinned로 2.7배
- Next: 품질 비교(8B full FT 학습 후 NSMC/MMLU) 또는 AdamW 고정비 줄이기; 카드 2장 분할(NPUTRAIN-18)
- Metrics: step_time_s=6.49
- Params: model=Llama-3.1-8B-Instruct, mode=full, seq_len=512, accum=4, device=rngd, a100_tok_s=476.9, rngd_tok_s=315.5, offload=resident-buffers
- Artifacts: experiments/data/buf8b-rngd.jsonl, experiments/data/full8b-cudabuf.jsonl
- Experiments: experiments/2026-10-16-full8b-offload-step.md
- By: claude@control-node · main@5ba96b2-dirty

### 2026-10-19 09:13+09:00 — NPUTRAIN-29 ✅

- Result: PR https://github.com/ai-computing/npu-lora/pull/1 — airm.workloads:npu-lora runner(npu_lora.airm:train), airm_tok_s/accuracy_pct, deploy/helm/npu-lora, ai-rm.yaml protocol 0.1. conformance: 기본 실행 PASS 3/FAIL 0, rm-core v0.1에 workloads 검사가 없어 그룹 지정 시 rm-core KeyError. test_airm 4/4, test_cpu 11/11, helm template OK, check_sensitive clean
- Next: PR 리뷰·머지; lmd-top collect.rs를 airm_tok_s(run_id)로 바꾼 뒤 재vendoring
- New todos: lmd-top: npu_lora_tokens_per_second → airm_tok_s(run_id 라벨)로 읽고 npu-lora 재vendoring; rm-core: conformance에 airm.workloads 검사 추가(지금은 KeyError), step_ms 지표 후보 제안
- By: claude@control-node · main@5ba96b2-dirty

### 2026-10-16 18:50+09:00 — NPUTRAIN-19

- Result: 8B full FT RNGD 1장 accum 4: 13.13 s/step(2048 tok) = 156 tok/s, accum 1(49.6 tok/s)의 3.1배. 전송 고정비 ~8.6 s는 그대로
- Next: A100 8B full FT 대조(e11 GPU 비면); 카드 2장 레이어 분할(NPUTRAIN-18)
- Metrics: step_time_s=13.13
- Params: model=Llama-3.1-8B-Instruct, mode=full, seq_len=512, accum=4, device=rngd, node=e11, offload=pipeline4
- Artifacts: experiments/data/pipe8b4-rngd.jsonl
- Experiments: experiments/2026-10-16-full8b-offload-step.md
- By: claude@control-node · main@5de85aa-dirty

### 2026-10-16 17:02+09:00 — NPUTRAIN-19

- Result: 8B full FT가 RNGD 1장에서 돈다: S=512 accum1 10.33 s/step(49.6 tok/s). 계산 ~1.5 s, 나머지는 기울기 14 GB D2H(3.1 GB/s)+가중치 H2D 전송 대기 — PCIe 병목
- Next: accum 4 그래프 컴파일(npu-node-1, ~2h) 후 step 측정; 2카드 레이어 분할로 링크 2개 활용
- Metrics: step_time_s=10.33
- Params: model=Llama-3.1-8B-Instruct, mode=full, seq_len=512, accum=1, device=rngd, node=e11, offload=pipeline4
- Artifacts: experiments/data/pipe8b-rngd.jsonl
- Experiments: experiments/2026-10-16-full8b-offload-step.md
- By: claude@control-node · main@5de85aa-dirty

### 2026-10-16 15:52+09:00 — NPUTRAIN-26 ✅

- Result: 민감정보 제거·ai-rm.yaml·README ai-rm 절 추가, checker clean, push ad02402. examples/cluster_facts.py IP 3개(NAS·게이트웨이·Prometheus)를 MODEL_STORE_NAS_HOST/LLMD_GATEWAY_HOST/PROMETHEUS_HOST/PORT 환경변수(cluster.local.env)로, cli.py serve 예시 IP를 <vllm-host>로 교체. 생성 데이터는 이전과 동일 확인
- Next: lmd-top 내 vendored 사본도 같은 수정 필요(다른 agent). 정적 k8s 매니페스트·전용 이미지 없음
- New todos: deploy/k8s 정적 Job 매니페스트(kustomize) 추가
- By: claude@control-node · main@ad02402-dirty

### 2026-10-16 15:07+09:00 — NPUTRAIN-17 ✅

- Result: replay 레시피 1B full FT: RNGD NSMC 78.2%/MMLU 36.25% vs A100 79.0%/36.75%, loss 상대차 1.2%. 과제만 학습 비교(79.8/20.25 vs 78.5/8.5)도 기록
- Next: NPUTRAIN-19: 8B full FT step 시간 측정
- Experiments: experiments/2026-10-16-full1b-nsmc-replay.md, experiments/2026-10-16-full1b-nsmc-task-only.md
- By: claude@control-node · main@0534900-dirty

### 2026-10-16 15:06+09:00 — 전체 파인튜닝 구현·검증

- Result: 1B full FT replay(NSMC+Alpaca 1:1): RNGD NSMC 78.2%/MMLU 36.25% vs A100 79.0%/36.75%, loss 곡선 상대차 1.2% — RNGD 전체 FT가 GPU와 같은 품질
- Next: 8B 전체 FT 컴파일 완료 후 e11 RNGD에서 오프로드 파이프라인 step 시간 측정
- Metrics: nsmc_accuracy_pct=78.2, mmlu_accuracy_pct=36.25, step_time_s=3.85
- Params: model=Llama-3.2-1B-Instruct, mode=full, seq_len=128, accum=4, steps=800, lr=1e-5, data=nsmc+alpaca_1to1, device=rngd
- Artifacts: experiments/data/fullmix-rngd.jsonl, experiments/data/fullmix-cuda.jsonl
- Experiments: experiments/2026-10-16-full1b-nsmc-replay.md
- By: claude@control-node · main@0534900-dirty

### 2026-10-16 14:10+09:00 — 8B full fine tuning과 A100 비교

- Result: 1B full FT(과제만): NSMC RNGD 79.8% / A100 78.5% (기본 62.8%), MMLU 20.25% / 8.5% (기본 43.25%) — 양쪽 같은 과적합 망각. RNGD 기울기 점검 PASS(최대 2.1%). replay(Alpaca 1:1) 재실험: RNGD NSMC 78.2%, MMLU 36.25%, A100 대조 진행 중
- Next: A100 replay 결과 비교, offload 파이프라인으로 8B full FT 측정
- Metrics: nsmc_accuracy_pct=78.2, mmlu_accuracy_pct=36.25
- Params: model=Llama-3.2-1B-Instruct, mode=full, data=nsmc+alpaca, lr=1e-5
- Artifacts: experiments/data/full-rngd.jsonl, experiments/data/full-cuda.jsonl
- Experiments: experiments/2026-10-16-full1b-nsmc-task-only.md
- By: claude@control-node · main@f232df7-dirty

### 2026-10-16 14:10+09:00 — LoRA 품질 검증 (RNGD vs A100) ✅

- Result: Llama-3.1-8B LoRA 같은 조건: NSMC test RNGD 87.5% vs A100 87.6% (기본 79.6%), MMLU 63.0% 둘 다 (기본 60.75%), loss 곡선 구간 상대차 중앙값 2.8%. llama3 rope_scaling 미적용·긴 예제 응답 잘림 버그 수정
- Next: 시드 여러 개 반복으로 분산 확인
- Metrics: nsmc_accuracy_pct=87.5, mmlu_accuracy_pct=63.0, step_time_s=2.6
- Params: model=Llama-3.1-8B-Instruct, device=rngd, seq_len=128, rank=16, lr=2e-4, steps=400, accum=4
- Artifacts: experiments/data/nsmc-rngd.jsonl, experiments/data/nsmc-cuda.jsonl
- Experiments: experiments/2026-10-16-nsmc-lora-rngd-vs-a100.md
- By: claude@control-node · main@f232df7-dirty

### 2026-10-16 11:47+09:00 — STATUS.md 초기 작성 ✅

- Result: 초안 검증·보완: README 수치 전부 일치 확인, full FT(--full, 1B 실증: NSMC 62.8%/MMLU 43.3%, step ~5 s, 망각→replay) 추가, RBLN 백엔드 부재·미커밋·experiments 기록 없음을 Open questions에 추가
- Next: 1B full fine-tuning을 RNGD와 A100에서 같은 조건으로 측정하고 experiments/에 기록, 8B full FT 다중 카드 분할 설계
- By: claude@control-node · main@f232df7-dirty

### 2026-10-16 11:43+09:00 — STATUS.md 초기 작성 ✅

- Result: STATUS.md 초안 작성(hub에서 저장소 스냅샷 기반, 노드 agent 검토 필요)
- Next: 1B full fine-tuning을 RNGD와 A100에서 같은 조건으로 측정해 비교 기준을 세우고, 8B full fine-tuning의 다중 카드 분할 방안을 설계한다.
- By: research-hub@control-node · main@f232df7-dirty
