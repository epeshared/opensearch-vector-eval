#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel

MODEL_NAME = "/home/xtang/models/Qwen/Qwen3-Embedding-4B"
TS_OUTPUT_PATH = "ts_models/qwen3_embedding_4b_ts_cpu.pt"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # 与官方示例一致
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


class Qwen3EmbeddingWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # 显式禁用 grad，避免 autograd/functorch 参与
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 norm
            return embeddings


def main():
    # 可选：确保没有 torch.compile / dynamo 干扰
    os.environ.setdefault("TORCH_DISABLE_FUNCTORCH_COMPAT", "1")

    device = torch.device("cpu")
    print(f"Loading model: {MODEL_NAME}")

    # 关键：强制使用 eager attention，而不是 SDPA
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",   # ✅ 避开 SDPA + masking_utils + vmap 那条路径
        torch_dtype=torch.float32,
    )
    base_model.eval().to(device)

    wrapper = Qwen3EmbeddingWrapper(base_model).eval().to(device)

    # dummy inputs（按你后面 Java 推理的 max_length 来设，这里先 16）
    batch_size = 2
    seq_len = 16
    example_input_ids = torch.ones(
        (batch_size, seq_len), dtype=torch.long, device=device
    )
    example_attention_mask = torch.ones(
        (batch_size, seq_len), dtype=torch.long, device=device
    )

    # 直接 trace，不再尝试 script（我们已经知道会炸在 **kwargs）
    print("Tracing wrapper with torch.jit.trace ...")
    scripted = torch.jit.trace(
        wrapper,
        (example_input_ids, example_attention_mask),
        check_inputs=[
            (
                torch.ones((1, seq_len), dtype=torch.long, device=device),
                torch.ones((1, seq_len), dtype=torch.long, device=device),
            )
        ],
        strict=False,
    )

    os.makedirs(os.path.dirname(TS_OUTPUT_PATH), exist_ok=True)
    scripted.save(TS_OUTPUT_PATH)
    print(f"Saved TorchScript model to: {TS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
