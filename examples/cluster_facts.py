"""데모 데이터: 원래 모델이 모르는 이 클러스터의 사실 10개.

사실마다 학습용 질문 표현 6개, 평가용 표현 1개(학습에 없음)를 만든다. 학습 뒤 평가 질문에
맞는 답(핵심어 포함)을 하면 어댑터가 사실을 익힌 것이다.

  python examples/cluster_facts.py out_dir   →  out_dir/train.jsonl, out_dir/eval.jsonl

사이트 고유 주소는 환경변수로 받는다(레포 루트의 cluster.local.env 가 있으면 읽는다,
cluster.example.env 참고): MODEL_STORE_NAS_HOST, LLMD_GATEWAY_HOST, PROMETHEUS_HOST, PROMETHEUS_PORT.
"""
import json
import os
import sys

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster.local.env")


def _load_env_file(path=ENV_FILE):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _need(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 없습니다. cluster.example.env 를 cluster.local.env 로 복사해 채우거나 export 하세요.")
    return v


_load_env_file()
NAS = _need("MODEL_STORE_NAS_HOST")
GATEWAY = _need("LLMD_GATEWAY_HOST")
PROM_HOST = _need("PROMETHEUS_HOST")
PROM_PORT = _need("PROMETHEUS_PORT")

FACTS = [
    ("A100 GPU", "ETRI llm-d 클러스터의 A100 80GB 두 장은 etri-esc8000-e11 노드에 있습니다.", "etri-esc8000-e11",
     ["A100은 어느 노드에 있어?", "A100 GPU가 꽂힌 서버 이름이 뭐야?", "우리 클러스터 A100 위치 알려줘",
      "A100 80GB는 어떤 노드에서 쓸 수 있나요?", "GPU A100이 설치된 호스트는?", "A100 카드가 있는 머신은 어디야?"],
     "A100 그래픽카드는 어느 장비에 장착돼 있지?"),
    ("lmd-top", "lmd-top은 llm-d 클러스터를 관측하고 운영하는 Rust 기반 터미널 도구입니다.", "Rust",
     ["lmd-top이 뭐야?", "lmd-top은 어떤 도구야?", "lmd-top 설명해줘", "lmd-top은 무슨 언어로 만들었어?",
      "lmd-top 용도가 뭐야?", "lmd-top은 어떤 프로그램인가요?"],
     "lmd-top이라는 툴에 대해 알려줄래?"),
    ("npu-lora", "npu-lora는 FuriosaAI RNGD 같은 추론용 NPU에서 LoRA 파인튜닝을 하는 라이브러리입니다.", "RNGD",
     ["npu-lora가 뭐야?", "npu-lora는 어떤 라이브러리야?", "npu-lora 설명해줘", "npu-lora로 무엇을 할 수 있어?",
      "npu-lora는 어떤 장비에서 학습해?", "npu-lora 용도는?"],
     "npu-lora 라이브러리는 무슨 일을 하나요?"),
    ("공유 스토어", f"모델 스토어는 NAS {NAS}의 lab 공유를 쓰는 model-store PVC이고, 여유 공간은 약 45TB입니다.",
     "model-store",
     ["모델 스토어는 어디에 있어?", "공유 모델 저장소 PVC 이름이 뭐야?", "모델 파일은 어디에 저장해?",
      "클러스터 공유 스토리지 알려줘", "어댑터 결과는 어디에 저장돼?", "NAS 모델 스토어 정보 알려줘"],
     "모델을 저장하는 공용 볼륨은 뭐라고 부르나요?"),
    ("RBLN", "etri-001 노드에는 Rebellions ATOM(RBLN-CA22) 네 장과 FuriosaAI RNGD 세 장이 있습니다.", "etri-001",
     ["ATOM NPU는 어느 노드에 있어?", "Rebellions 카드가 있는 서버는?", "RBLN-CA22는 몇 장이고 어디 있어?",
      "etri-001에는 어떤 가속기가 있어?", "ATOM이 설치된 노드 알려줘", "리벨리온 NPU 위치가 어디야?"],
     "Rebellions ATOM 카드는 어떤 머신에 꽂혀 있나요?"),
    ("게이트웨이", f"llm-d 게이트웨이의 사내 접속 주소는 {GATEWAY}입니다.", GATEWAY,
     ["llm-d 게이트웨이 주소가 뭐야?", "추론 요청은 어디로 보내?", "게이트웨이 IP 알려줘",
      "클러스터 LLM 엔드포인트 주소는?", "llm-d 진입점 IP가 뭐야?", "모델 API 접속 주소 알려줘"],
     "LLM 서비스에 요청을 보낼 사내 IP는 무엇인가요?"),
    ("Prometheus", f"클러스터 Prometheus는 {PROM_HOST}:{PROM_PORT}에서 접속합니다.", PROM_PORT,
     ["Prometheus 주소가 뭐야?", "메트릭 서버 어디야?", "프로메테우스 접속 포트 알려줘",
      "모니터링 Prometheus URL은?", "Prometheus는 몇 번 포트야?", "지표 수집 서버 주소 알려줘"],
     "Prometheus 대시보드에 들어가려면 어디로 가야 해?"),
    ("DGX Spark", "GB10 노드 두 대(dgx-spark0, gx10-4744)는 디스크가 차면 응답이 끊기니 디스크를 15% 이상 비워 둬야 합니다.",
     "디스크",
     ["GB10 노드가 자주 죽는 이유가 뭐야?", "dgx-spark0이 멈추는 원인은?", "gx10-4744 장애 원인 알려줘",
      "DGX Spark 노드 관리할 때 주의할 점은?", "GB10이 NotReady 되는 이유는?", "스파크 노드가 끊기는 이유가 뭐야?"],
     "GB10 장비가 응답하지 않게 되는 건 왜 그래?"),
    ("RNGD 컴파일", "RNGD 컴파일러는 큰 그래프에서 느려지므로 npu-lora는 그래프를 행렬곱 두 개 단위로 잘라 컴파일합니다.",
     "행렬곱",
     ["npu-lora는 왜 그래프를 나눠?", "RNGD에서 학습 그래프를 어떻게 컴파일해?", "npu-lora 그래프 분할 이유는?",
      "학습 그래프 컴파일이 느릴 때 어떻게 해?", "npu-lora 컴파일 전략 알려줘", "왜 그래프를 조각내서 컴파일해?"],
     "npu-lora가 컴파일할 때 그래프를 쪼개는 이유는 뭔가요?"),
    ("측정", "이종 가속기 에너지는 노드 전체가 아니라 배정된 카드 전력을 1초마다 적분해 J/token으로 잽니다.", "J/token",
     ["에너지는 어떻게 측정해?", "가속기 전력 측정 방법 알려줘", "에너지 효율 지표가 뭐야?",
      "이종 가속기 실험에서 전력은 어떻게 재?", "카드 전력 측정 방식은?", "에너지 측정 단위가 뭐야?"],
     "이종 가속기 실험의 에너지 지표는 어떻게 계산하나요?"),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        for _, ans, _, qs, _ in FACTS:
            for q in qs:
                f.write(json.dumps({"prompt": q, "response": ans}, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "eval.jsonl"), "w") as f:
        for name, ans, key, _, q in FACTS:
            f.write(json.dumps({"fact": name, "prompt": q, "expect": key, "answer": ans}, ensure_ascii=False) + "\n")
    print(f"{len(FACTS) * 6} train, {len(FACTS)} eval -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cluster_facts")
