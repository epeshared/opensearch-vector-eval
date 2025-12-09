#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

TS_PATH = "/mnt/nvme2n1p1/xtang/opensearch-playground/opensearch-vector-eval/src/embedding/ts_models/sentence-transformers/msmarco_distilbert_base_ts.pt"
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-tas-b"

texts = [
    "hello world",
    "this is a sentence for testing embedding",
]

def encode_hf(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        batch = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        out = model(**batch)
        # 和导出脚本一致：mean pooling + L2
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        masked = hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / lengths
        emb = F.normalize(pooled, p=2, dim=1)
        return emb

def encode_ts(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ts = torch.jit.load(TS_PATH)
    ts.eval()

    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = ts(batch["input_ids"], batch["attention_mask"])
    return emb

if __name__ == "__main__":
    emb_hf = encode_hf(texts)
    emb_ts = encode_ts(texts)

    print("HF emb[0][:8]:", emb_hf[0][:8])
    print("TS emb[0][:8]:", emb_ts[0][:8])

    cos = F.cosine_similarity(emb_hf, emb_ts, dim=-1)
    print("cos(HF, TS) per sample:", cos)
