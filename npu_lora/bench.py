"""품질 평가: vLLM 오프라인 엔진으로 기본 모델·LoRA 어댑터(·전체 파인튜닝 모델)를 같은 조건에서 채점한다.

vLLM 이미지(GPU)에서 돈다. 한 번 실행에 모델 하나, 그 위에 LoRA 어댑터 여러 개.

  python -m npu_lora.bench --model meta-llama/Llama-3.1-8B-Instruct \
      --adapter npu=/mnt/store/adapters/.../nsmc-rngd --adapter gpu=/mnt/store/adapters/.../nsmc-cuda \
      --task nsmc=nsmc_test.jsonl --task mmlu=mmlu.jsonl --out result.json

과제 JSONL: {"prompt", "expect"}. 채점: nsmc 는 답에 처음 나오는 '긍정'/'부정' 이 정답인지,
mmlu 는 답에 처음 나오는 A–D 글자가 정답인지(생성 길이 8, temperature 0).
"""
import argparse
import json
import re
import time


def score(task, answer, expect):
    if task.startswith("nsmc"):
        m = re.search(r"(긍정|부정)", answer)
        return bool(m and m.group(1) == expect)
    m = re.search(r"\b([ABCD])\b", answer) or re.search(r"([ABCD])", answer)
    return bool(m and m.group(1) == expect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=path (PEFT LoRA)")
    ap.add_argument("--task", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="base", help="이 모델(어댑터 없음)의 이름")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapters = [x.split("=", 1) for x in a.adapter]
    llm = LLM(model=a.model, enable_lora=bool(adapters), max_lora_rank=a.max_lora_rank, max_loras=max(1, len(adapters)),
              max_model_len=2048, gpu_memory_utilization=a.gpu_mem, enforce_eager=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=8)
    variants = [(a.label, None)] + [(n, LoRARequest(n, i + 1, p)) for i, (n, p) in enumerate(adapters)]
    results = {}
    for tname, tpath in (x.split("=", 1) for x in a.task):
        rows = [json.loads(l) for l in open(tpath)]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                           add_generation_prompt=True) for r in rows]
        for vname, lora in variants:
            t0 = time.time()
            outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
            answers = [o.outputs[0].text for o in outs]
            ok = [score(tname, ans, r["expect"]) for ans, r in zip(answers, rows)]
            results.setdefault(vname, {})[tname] = {
                "accuracy": sum(ok) / len(ok), "n": len(ok), "seconds": round(time.time() - t0, 1),
                "samples": [{"prompt": r["prompt"][-80:], "expect": r["expect"], "answer": ans.strip()[:40]}
                            for r, ans in list(zip(rows, answers))[:5]]}
            print(json.dumps({"variant": vname, "task": tname, "accuracy": results[vname][tname]["accuracy"]}),
                  flush=True)
    json.dump({"model": a.model, "results": results}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
