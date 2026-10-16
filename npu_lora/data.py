"""학습 데이터: JSONL → 길이 S 로 묶은 (ids, labels, doc_ids).

한 줄은 {"text": ...} 또는 {"prompt": ..., "response": ...} 또는 {"messages": [...]}.
prompt/messages 형식은 응답 토큰에만 loss 를 준다. 짧은 예제는 여러 개를 한 시퀀스에 묶고
(packing), 문서 경계를 넘지 않는 블록 대각 인과 마스크를 쓰도록 doc_ids 를 함께 돌려준다 —
마스크가 그래프 입력이라 컴파일을 다시 하지 않아도 된다.
"""
import json
import random

import torch


def _encode(tok, ex):
    """→ (token ids, 각 토큰이 loss 대상인지)"""
    if "text" in ex:
        ids = tok(ex["text"], add_special_tokens=True)["input_ids"]
        return ids, [True] * len(ids)
    if "messages" in ex:
        msgs = ex["messages"]
        prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tok.apply_chat_template(msgs, tokenize=True)
    else:
        msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        full_ids = tok.apply_chat_template(msgs + [{"role": "assistant", "content": ex["response"]}],
                                           tokenize=True)
    prompt_ids, full_ids = _ids(prompt_ids), _ids(full_ids)
    n = len(prompt_ids)
    return full_ids, [False] * n + [True] * (len(full_ids) - n)


def _fit(ids, train, n):
    """길이 n 으로 줄인다. 응답(학습 대상) 토큰은 지키고 그 바로 앞 프롬프트를 덜어낸다.

    뒤를 자르면 응답이 통째로 사라져 그 예제는 학습에 아무 기여도 못 한다(NSMC 4000개 중 291개).
    응답만으로도 n 을 넘으면 그때는 뒤를 자른다.
    """
    excess = len(ids) - n
    first = next((i for i, t in enumerate(train) if t), len(ids))
    if first - excess >= 1:                       # 프롬프트 앞머리(템플릿·지시문)는 남기고 끝부분을 덜어낸다
        cut = first - excess
        return ids[:cut] + ids[first:], train[:cut] + train[first:]
    return ids[:n], train[:n]


def _ids(x):
    # transformers 5 는 BatchEncoding 을, 4 는 list 를 돌려준다
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)


class PackedDataset:
    def __init__(self, path, tokenizer, seq_len, seed=0):
        self.S = seq_len
        self.examples = []
        dropped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids, train = _encode(tokenizer, json.loads(line))
                if len(ids) < 2:
                    continue
                if len(ids) > seq_len + 1:
                    ids, train = _fit(ids, train, seq_len + 1)
                    dropped += 1
                self.examples.append((ids, train))
        self.truncated = dropped
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.examples)

    def sequences(self):
        """한 epoch 분량을 섞어서 S 길이로 묶어 낸다."""
        order = list(range(len(self.examples)))
        self.rng.shuffle(order)
        ids, lab, doc = [], [], []
        d = 0
        for i in order:
            x, train = self.examples[i]
            # 다음 토큰 예측: 입력 x[:-1], 정답 x[1:] (정답 토큰이 학습 대상일 때만)
            xi, yi = x[:-1], [t if m else -1 for t, m in zip(x[1:], train[1:])]
            if len(ids) + len(xi) > self.S and ids:
                yield self._finish(ids, lab, doc)
                ids, lab, doc, d = [], [], [], 0
            ids += xi
            lab += yi
            doc += [d] * len(xi)
            d += 1
        if ids:
            yield self._finish(ids, lab, doc)

    def _finish(self, ids, lab, doc):
        n = self.S - len(ids)
        ids = ids + [self.pad] * n
        lab = lab + [-1] * n
        doc = doc + [doc[-1] + 1 if doc else 0] * n     # 패딩은 별도 문서 — 실제 토큰에 영향 없음
        return torch.tensor(ids), torch.tensor(lab), torch.tensor(doc)

    def batches(self, accum, steps):
        """steps 개의 [(ids, labels, doc_ids)] × accum — 필요하면 epoch 를 여러 번 돈다."""
        out = []
        while True:
            for s in self.sequences():
                out.append(s)
                if len(out) == accum:
                    yield out
                    out = []
                    steps -= 1
                    if steps == 0:
                        return
