#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用文本 embedding 模型 TorchScript 导出脚本

示例用法：

1）导出 BGE 大模型：
    python export_text_embedding_ts.py \
        --model-name BAAI/bge-large-en-v1.5 \
        --output ts_models/bge_large_en_ts.pt \
        --pooling mean --max-seq-len 128

2）导出 Qwen3-Embedding-4B（注意：当前环境下可能因为 transformers 内部实现导致失败）：
    python export_text_embedding_ts.py \
        --model-name /home/xtang/models/Qwen/Qwen3-Embedding-4B \
        --output ts_models/qwen3_embedding_4b_ts_cpu.pt \
        --pooling last --max-seq-len 128 --attn-eager
"""

import os
import argparse
from typing import Literal, Dict  # ★ 多了 Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


PoolType = Literal["mean", "cls", "last"]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """与你原来的实现一致：支持左 padding / 右 padding 的 last-token pooling。"""
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


class TextEmbeddingWrapper(torch.nn.Module):
    """
    通用 embedding 包装器：

    ⚠ 关键点：为了适配 OpenSearch ML，
    forward 接收一个 Dict[str, Tensor]，而不是两个 Tensor 参数。
    OpenSearch 侧会传入类似：
        {
          "input_ids": Tensor[int64] [B, L],
          "attention_mask": Tensor[int64] [B, L]
        }
    """

    POOL_MEAN = 0
    POOL_CLS = 1
    POOL_LAST = 2

    def __init__(self, base_model: torch.nn.Module, pooling: PoolType, dtype: str = "bfloat16"):
        super().__init__()
        self.base_model = base_model

        if dtype == "bfloat16" or dtype == "bf16":
            self.base_model.to(torch.bfloat16)
        elif dtype == "float16" or dtype == "fp16":
            self.base_model.to(torch.float16)

        if pooling == "mean":
            self.pooling_type = self.POOL_MEAN
        elif pooling == "cls":
            self.pooling_type = self.POOL_CLS
        else:
            # "last"
            self.pooling_type = self.POOL_LAST

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        inputs: {"input_ids": Tensor, "attention_mask": Tensor}
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 显式禁用 grad，避免 autograd/functorch 开销
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state  # [B, L, H]

            if self.pooling_type == self.POOL_MEAN:
                # mean pooling：对 padding token 屏蔽
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
                masked_hidden = hidden * mask
                sum_hidden = masked_hidden.sum(dim=1)  # [B, H]
                lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
                pooled = sum_hidden / lengths

            elif self.pooling_type == self.POOL_CLS:
                # CLS pooling：取第一个 token
                pooled = hidden[:, 0]

            else:
                # last-token pooling（兼容左/右 padding）
                pooled = last_token_pool(hidden, attention_mask)

            embeddings = F.normalize(pooled, p=2, dim=1)
            return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HuggingFace text embedding model to TorchScript"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save TorchScript model, e.g. ts_models/model_ts.pt",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["mean", "cls", "last"],
        help="Pooling strategy: mean / cls / last_token (default: last).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Sequence length used for dummy tracing inputs (and later inference).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Dummy batch size used for tracing (仅影响导出，不影响推理接口).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16","float32"],
        help="Model dtype when loading (default: bfloat16).",
    )
    parser.add_argument(
        "--attn-eager",
        action="store_true",
        help="Use attn_implementation='eager' when loading model "
             "(对部分 decoder-only 模型如 Qwen3 有帮助).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel.from_pretrained.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 可选：关闭一些 functorch 兼容逻辑（对导出有时更干净）
    os.environ.setdefault("TORCH_DISABLE_FUNCTORCH_COMPAT", "1")

    device = torch.device("cpu")
    print(f"[Config] model_name={args.model_name}")
    print(f"[Config] output={args.output}")
    print(f"[Config] pooling={args.pooling}, max_seq_len={args.max_seq_len}, "
          f"batch_size={args.batch_size}, dtype={args.dtype}")
    if args.attn_eager:
        print("[Config] attn_implementation='eager' enabled")
    if args.trust_remote_code:
        print("[Config] trust_remote_code=True")

    # 映射 dtype
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    

    # 构建 from_pretrained 的 kwargs，兼容 Qwen3、BGE 等
    from_pretrained_kwargs = dict(
        dtype=dtype,
        device_map=None,  # 明确等会儿我们自己 .to(device)
    )
    if args.attn_eager:
        from_pretrained_kwargs["attn_implementation"] = "eager"
    if args.trust_remote_code:
        from_pretrained_kwargs["trust_remote_code"] = True

    print(f"[Load] Loading model: {args.model_name}")
    base_model = AutoModel.from_pretrained(
        args.model_name,
        **from_pretrained_kwargs,
    )
    base_model.eval().to(device)

    # 包装成通用 embedding 模型
    wrapper = TextEmbeddingWrapper(base_model, pooling=args.pooling, dtype=args.dtype).eval().to(device)

    # 构造 dummy 输入用于 script/trace —— 注意是 dict
    batch_size = args.batch_size
    seq_len = args.max_seq_len

    example_inputs = {
        "input_ids": torch.ones(
            (batch_size, seq_len), dtype=torch.long, device=device
        ),
        "attention_mask": torch.ones(
            (batch_size, seq_len), dtype=torch.long, device=device
        ),
    }

    print("[Export] Trying torch.jit.script(wrapper) first ...")
    scripted = None
    try:
        scripted = torch.jit.script(wrapper)
        # 触发一下运行，确保脚本化路径没问题
        _ = scripted(example_inputs)
        print("[Export] torch.jit.script succeeded.")
    except Exception as e:
        print("[Warn] torch.jit.script failed, fallback to trace:")
        print(e)
        print("[Export] Tracing wrapper with torch.jit.trace ...")
        scripted = torch.jit.trace(
            wrapper,
            (example_inputs,),
            strict=False,
        )

    # 保存 TorchScript 模型
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    scripted.save(args.output)
    print(f"[Done] Saved TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
