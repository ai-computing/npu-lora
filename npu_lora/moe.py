"""MoE 체크포인트의 MLP 절반(post-attention norm + MoE 블록)을 host 에서 고정한 채 돌린다.

furiosa-torch 에는 장치 쪽 gather 가 없어 token→expert 분배를 컴파일된 그래프 하나로 만들 수 없다. 그래서 MoE
레이어를 나눈다: attention + residual 은 장치 그래프(`llama.layer`, `LlamaConfig.moe`), 그 뒤 norm + MoE 블록은
여기서 HF 모듈 그대로(라우팅·shared expert·가중치 변환이 계열마다 HF 와 같다). expert 는 고정이고, 역전파는
입력(attention 출력)에 대한 기울기만 낸다 — LoRA 는 attention 투영에 붙는다.
"""
import torch

BF = torch.bfloat16


class HostMLP:
    """레이어 i 의 (post_attention_layernorm, mlp) 를 HF 모델에서 가져와 host 에 둔다(bf16, requires_grad=False).

    forward(i, a)  = a + mlp(norm(a))            a: [S, H] host bf16 (attention + residual 출력)
    backward(i, a, g) = d forward(i, a) / d a · g  (순전파를 다시 계산한다)
    """

    def __init__(self, path, dtype=BF):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        self.blocks = []
        for layer in model.model.layers:
            norm, mlp = layer.post_attention_layernorm, layer.mlp
            norm.requires_grad_(False)
            mlp.requires_grad_(False)
            self.blocks.append((norm, mlp))
        del model                      # embeddings, attention and lm_head are freed; the blocks stay

    def _f(self, i, a):
        norm, mlp = self.blocks[i]
        y = mlp(norm(a.unsqueeze(0)))
        if isinstance(y, tuple):       # some MoE blocks also return router logits
            y = y[0]
        return a + y.reshape(a.shape).to(a.dtype)

    def forward(self, i, a):
        with torch.no_grad():
            return self._f(i, a)

    def backward(self, i, a, g):
        x = a.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self._f(i, x)
            (gx,) = torch.autograd.grad(y, x, g.to(y.dtype))
        return gx
