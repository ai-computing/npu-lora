# STATUS — RNGD NPU 파인튜닝 학습

<!-- research-hub:header:begin (maintained by .research/report.py; keep the "- **Key**: value" format) -->
- **State**: active
- **Updated**: 2026-10-16 14:10+09:00
- **Agent**: claude@gspark-PowerEdge-R7525
- **Task**: 8B full fine tuning과 A100 비교
- **Last result**: 1B full FT(과제만): NSMC RNGD 79.8% / A100 78.5% (기본 62.8%), MMLU 20.25% / 8.5% (기본 43.25%) — 양쪽 같은 과적합 망각. RNGD 기울기 점검 PASS(최대 2.1%). replay(Alpaca 1:1) 재실험: RNGD NSMC 78.2%, MMLU 36.25%, A100 대조 진행 중
- **Next**: A100 replay 결과 비교, offload 파이프라인으로 8B full FT 측정
- **Blockers**: none
<!-- research-hub:header:end -->

## Goal

추론 전용 AOT 컴파일 NPU인 Furiosa RNGD에서 LoRA와 full fine-tuning을 실용 수준으로 돌리고, 같은 조건의 A100 학습과 품질·속도를 비교한다. 구현은 llm-d 레포의 `npu-lora/` 패키지(`npu_lora`, 백엔드 furiosa/cuda/cpu)다. 결과를 llm-d 서빙(vLLM `--enable-lora`)과 lmd-top(`--plan train`)에 붙이는 것까지 범위에 넣는다 (`npu-lora/README.md`).

## Findings

최신순. 출처를 괄호에 적었다. 레포에 원자료가 없는 수치는 그렇다고 밝혔다.

- **2026-10-16 · RNGD LoRA 학습 품질이 A100 학습과 같다.** Llama-3.1-8B, S=128, r=16 q·v, lr 2e-4, 400 step × accum 4, 같은 데이터와 시드로 비교했다. NSMC 시험(n=1000)은 기본 79.6%, RNGD 87.5%, A100 87.6%였다. MMLU(n=400)는 기본 60.8%, RNGD와 A100 모두 63.0%였다. loss 곡선 40 step 구간 평균의 상대차 중앙값은 2.8%, 마지막 구간은 0.0676 대 0.0667이었다. 평가는 `npu_lora.bench`(vLLM 오프라인 엔진, temperature 0), 곡선 비교는 `npu_lora.compare`로 했다 (`README.md` 「품질 검증」).
- **2026-10-16 · 품질 검증 중 버그 2건을 고쳤다.** llama3 `rope_scaling`이 적용되지 않았다(HF 대비 오차 1.8e-4 → 3.5e-7). 긴 예제는 응답째 잘렸다(이제 프롬프트 쪽을 덜어낸다) (`README.md`).
- **full fine-tuning 모드가 있고 1B로만 실증했다.** `--full`(lmd-top `--set full=true`)은 레이어 가중치 9종을 학습하고 임베딩·lm_head는 고정한다. 결과는 HF 전체 모델로 저장한다 (`npu_lora/cli.py`, `trainer.py`, `export.py` `save_full`). 8B는 128 GB가 필요해 카드 1장으로 할 수 없어서 `unsloth/Llama-3.2-1B-Instruct`로 실증했다. 1B 기본 성능은 NSMC 62.8%, MMLU 43.3%였다. RNGD 기울기 점검은 최대 2.1%로 통과했고, step은 약 5 s였다(대부분 9.7억 파라미터 host AdamW). 이 수치들은 2026-10-15 작업 세션 기록이고 레포에 원자료는 없다. 한 과제로만 full FT를 하면 모든 질문에 그 과제 형식으로 답하는 망각이 생겼다(1B MMLU 43% → 8~20%). 그래서 범용 지시문(Alpaca)을 1:1로 섞는 replay 데이터를 만들었다 (`examples/nsmc.py`). RNGD 대 A100 full FT 비교 수치는 아직 없다.
- **2026-10-15 · RNGD 1장으로 Llama-3.1-8B 32층 LoRA 학습 PoC를 마쳤다.** 0.7 s/step(약 180 tok/s)였고 200 step에 2분 36초, loss 3.29 → ~0이었다. 학습에 없는 질문은 0/10에서 10/10이 됐다(A100 vLLM 런타임 LoRA로 평가, CUDA 백엔드 학습도 10/10). RNGD 대 CPU 기울기는 cos ≥ 0.9997, 상대오차 ≤ 2.5%(Llama-3.1-8B 2층 + 128k 어휘 head, loss 1.9e-4)였다. 함수형 Llama 대 HF loss 상대오차는 1.7e-5였다 (`README.md` 「검증 결과」).
- **컴파일 비용과 구조.** 첫 컴파일에 약 1.5시간이 걸리고, 이후에는 EDF 캐시(15 MB)로 바로 시작한다. Llama-8B 레이어는 순전파 8조각, 역전파 19조각으로 컴파일해 32층과 모든 어휘 조각에 재사용한다. 어댑터(r=16, q·v)는 13.6 MB라 host AdamW(fp32) 왕복 비용이 작다 (`README.md` 「왜 NPU 에서 학습이 되나」).
- **furiosa-torch 2026.2 컴파일러 우회를 14종 썼다.** 그중 3종은 "조용히 틀린 값"을 낸다: 조각 출력 전치, 컴파일 샘플 입력 stride 고정, make_fx 중복 입력 병합. 이 경우는 컴파일 오류로 드러나지 않아 `npu_lora.diagnose.compare_parts`로 조각마다 CPU와 비교해 잡는다 (`README.md` 「컴파일러 우회」).
- **llm-d·lmd-top 연동.** 출력은 HF PEFT 형식이라 vLLM `--enable-lora`와 llm-d LoRA 라우팅이 그대로 읽는다. `npu-lora job`과 `lmd-top --plan train`으로 클러스터 Job을 만든다(lmd-top 쪽 연동은 미커밋, `lmd-top/STATUS.md`). `--metrics-port`로 Prometheus 지표(`npu_lora_loss`, `npu_lora_tokens_per_second` 등)를 내고, `npu-lora job`이 ServiceMonitor도 만든다 (`README.md` 「사용」「관측」). CPU 회귀 테스트는 `tests/test_cpu.py`에 10개 있다.

## Open questions

- **8B full fine-tuning을 어떻게 나눌지 정하지 않았다.** 128 GB가 필요해 여러 RNGD 카드에 나눠야 한다. A100 대비 속도·품질 비교 조건도 미정이다. 1B full FT의 RNGD 대 A100 같은 조건 비교가 먼저다.
- **첫 컴파일에 1.5시간이 든다.** 우회 패스가 다른 모델과 SDK(예: Furiosa 2026.4)에서도 일반화되는지 확인하지 않았다 (`README.md`).
- **지원 범위가 좁다.** 지금은 Llama 계열(GQA, RoPE, SwiGLU)만 된다. micro-batch는 시퀀스 1개이고, 시퀀스 길이는 컴파일 시점에 고정된다 (`README.md` 「한계」).
- **RBLN(ATOM) 백엔드가 없다.** 같은 방식(`rebel.compile_from_torch`)이 가능하지만 이 클러스터의 RBLN 컴파일 경로가 고장 나 있어 넣지 않았다 (`README.md` 「한계」, `lmd-top/docs/RBLN-COMPILE-INCIDENT.md`).
- **Job 구성에 제약이 있다.** furiosa-torch는 privileged 컨테이너에서만 장치를 연다. 그래서 init 컨테이너가 배정 카드를 적고 학습 컨테이너가 서수(`rngd:N`)를 쓴다 (`README.md`).
- **새 모델마다 검증이 필요하다.** 조용히 틀리는 컴파일 결과는 오류로 드러나지 않으므로 `npu-lora check`와 `diagnose.compare_parts`를 매번 돌려야 한다.
- **`npu-lora/`가 아직 레포에 커밋되지 않았다**(`git status`: `?? npu-lora/`). `experiments/`에는 템플릿뿐이라 위 측정의 실험 기록이 없다.

## Log

<!-- research-hub:log (report.py inserts new entries below this line, newest first) -->

### 2026-10-16 14:10+09:00 — 8B full fine tuning과 A100 비교

- Result: 1B full FT(과제만): NSMC RNGD 79.8% / A100 78.5% (기본 62.8%), MMLU 20.25% / 8.5% (기본 43.25%) — 양쪽 같은 과적합 망각. RNGD 기울기 점검 PASS(최대 2.1%). replay(Alpaca 1:1) 재실험: RNGD NSMC 78.2%, MMLU 36.25%, A100 대조 진행 중
- Next: A100 replay 결과 비교, offload 파이프라인으로 8B full FT 측정
- Metrics: nsmc_accuracy_pct=78.2, mmlu_accuracy_pct=36.25
- Params: model=Llama-3.2-1B-Instruct, mode=full, data=nsmc+alpaca, lr=1e-5
- Artifacts: experiments/data/full-rngd.jsonl, experiments/data/full-cuda.jsonl
- Experiments: experiments/2026-10-16-full1b-nsmc-task-only.md
- By: claude@gspark-PowerEdge-R7525 · main@f232df7-dirty

### 2026-10-16 14:10+09:00 — LoRA 품질 검증 (RNGD vs A100) ✅

- Result: Llama-3.1-8B LoRA 같은 조건: NSMC test RNGD 87.5% vs A100 87.6% (기본 79.6%), MMLU 63.0% 둘 다 (기본 60.75%), loss 곡선 구간 상대차 중앙값 2.8%. llama3 rope_scaling 미적용·긴 예제 응답 잘림 버그 수정
- Next: 시드 여러 개 반복으로 분산 확인
- Metrics: nsmc_accuracy_pct=87.5, mmlu_accuracy_pct=63.0, step_time_s=2.6
- Params: model=Llama-3.1-8B-Instruct, device=rngd, seq_len=128, rank=16, lr=2e-4, steps=400, accum=4
- Artifacts: experiments/data/nsmc-rngd.jsonl, experiments/data/nsmc-cuda.jsonl
- Experiments: experiments/2026-10-16-nsmc-lora-rngd-vs-a100.md
- By: claude@gspark-PowerEdge-R7525 · main@f232df7-dirty

### 2026-10-16 11:47+09:00 — STATUS.md 초기 작성 ✅

- Result: 초안 검증·보완: README 수치 전부 일치 확인, full FT(--full, 1B 실증: NSMC 62.8%/MMLU 43.3%, step ~5 s, 망각→replay) 추가, RBLN 백엔드 부재·미커밋·experiments 기록 없음을 Open questions에 추가
- Next: 1B full fine-tuning을 RNGD와 A100에서 같은 조건으로 측정하고 experiments/에 기록, 8B full FT 다중 카드 분할 설계
- By: claude@gspark-PowerEdge-R7525 · main@f232df7-dirty

### 2026-10-16 11:43+09:00 — STATUS.md 초기 작성 ✅

- Result: STATUS.md 초안 작성(hub에서 저장소 스냅샷 기반, 노드 agent 검토 필요)
- Next: 1B full fine-tuning을 RNGD와 A100에서 같은 조건으로 측정해 비교 기준을 세우고, 8B full fine-tuning의 다중 카드 분할 방안을 설계한다.
- By: research-hub@gspark-PowerEdge-R7525 · main@f232df7-dirty
