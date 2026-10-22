"""npu-train (패키지 npu_train, 예전 이름 npu_lora): AOT 컴파일 NPU(FuriosaAI RNGD 등)에서 LoRA·전체 파인튜닝.

추론 전용으로 나온 NPU 에서 학습이 되는 이유: 고정 가중치와 LoRA 어댑터를 모두 그래프 입력으로 두고,
순전파+역전파를 make_fx 로 펼친 그래프 하나를 컴파일한다. 옵티마이저는 host 에서 돈다.
"""
__version__ = "0.1.0"

_LAZY = {"LoRAConfig", "LoRATrainer", "TrainConfig"}


def __getattr__(name):
    # torch 없이도 `npu-lora job`(매니페스트 출력)이 돌도록 학습 모듈은 쓸 때 불러온다.
    if name in _LAZY:
        from . import trainer
        return getattr(trainer, name)
    raise AttributeError(name)
