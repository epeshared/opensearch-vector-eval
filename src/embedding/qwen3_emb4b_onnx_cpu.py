#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel

MODEL_NAME = "/home/xtang/models/Qwen/Qwen3-Embedding-4B"
ONNX_OUTPUT_PATH = "onnx_models/qwen3_embedding_4b_cpu.onnx"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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
        # 导出时不需要 grad
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


def main():
    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

    device = torch.device("cpu")
    print(f"Loading model: {MODEL_NAME}")

    # 用 dtype 而不是 torch_dtype（去掉 warning）
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",   # 仍然用 eager，避开 SDPA 那套
        dtype=torch.float32,
    )
    base_model.eval().to(device)

    wrapper = Qwen3EmbeddingWrapper(base_model).eval().to(device)

    # 固定一个你打算在线上使用的最大 seq_len，例如 128
    batch_size = 1
    seq_len = 128   # <<< 按你实际需求改，比如 128/256

    example_input_ids = torch.ones(
        (batch_size, seq_len), dtype=torch.long, device=device
    )
    example_attention_mask = torch.ones(
        (batch_size, seq_len), dtype=torch.long, device=device
    )

    print("Exporting to ONNX (dynamo=False, fixed seq_len) ...")

    torch.onnx.export(
        wrapper,
        (example_input_ids, example_attention_mask),
        ONNX_OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        # 不要 dynamic_axes，保持静态 shape，避免 dynamo 相关约束
        do_constant_folding=True,
        opset_version=18,   # 直接 18，别管 17 了
        dynamo=False,       # <<< 关键：禁用 torch.export/dynamo 路径
    )

    print(f"Saved ONNX model to: {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
