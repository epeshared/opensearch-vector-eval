#!/usr/bin/env python3
"""Compare Yahoo embedding files by matching on the query text."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

DTYPE_ALIASES = {
    "FP16": "FLOAT16",
    "F16": "FLOAT16",
    "HALF": "FLOAT16",
    "FP32": "FLOAT32",
    "F32": "FLOAT32",
    "FLOAT": "FLOAT32",
    "FP64": "FLOAT64",
    "F64": "FLOAT64",
    "DOUBLE": "FLOAT64",
}

DTYPE_NAME_TO_NUMPY = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}


def normalize_dtype_label(label: Optional[str]) -> str:
    if not label:
        return "UNKNOWN"
    cleaned = label.strip().upper().replace(" ", "")
    return DTYPE_ALIASES.get(cleaned, cleaned)


def resolve_numpy_dtype(label: str, fallback: np.dtype) -> np.dtype:
    np_type = DTYPE_NAME_TO_NUMPY.get(label)
    return np.dtype(np_type) if np_type is not None else fallback


def to_target_dtype(values: Iterable[float], src_label: str, target_dtype: np.dtype) -> np.ndarray:
    src_dtype = resolve_numpy_dtype(src_label, target_dtype)
    arr = np.asarray(list(values), dtype=src_dtype)
    arr = np.reshape(arr, (-1,))
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)
    return arr


def load_embeddings_by_query(
    path: Path, target_dtype: np.dtype
) -> Tuple[Dict[str, np.ndarray], Counter, int]:
    data: Dict[str, np.ndarray] = {}
    dtype_hist: Counter = Counter()
    duplicate_queries = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("query")
            embedding = obj.get("embedding")
            dtype_label = normalize_dtype_label(obj.get("data_type"))

            if query is None or embedding is None:
                continue

            dtype_hist[dtype_label] += 1

            vector = to_target_dtype(embedding, dtype_label, target_dtype)

            if query in data:
                duplicate_queries += 1
                continue

            data[query] = vector

    return data, dtype_hist, duplicate_queries


def cosine_and_l2(vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[float, float]:
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector dim mismatch: {vec_a.shape} vs {vec_b.shape}")

    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.dot(vec_a, vec_a))
    norm_b = float(np.dot(vec_b, vec_b))
    diff = vec_a - vec_b
    l2 = float(np.linalg.norm(diff))

    if norm_a == 0.0 or norm_b == 0.0:
        cosine = float("nan")
    else:
        cosine = dot / math.sqrt(norm_a * norm_b)

    return cosine, l2


def summarize(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    v_min = min(values)
    v_max = max(values)
    v_mean = sum(values) / len(values)
    return v_mean, v_min, v_max


def render_histogram(counter: Counter) -> str:
    if not counter:
        return "(none)"
    parts = [f"{key}:{counter[key]}" for key in sorted(counter.keys())]
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embeddings by query text.")
    parser.add_argument("--file-a", type=Path, required=True, help="First embeddings JSONL file")
    parser.add_argument("--file-b", type=Path, required=True, help="Second embeddings JSONL file")
    parser.add_argument("--top-k", type=int, default=10, help="Show top-K queries by L2 distance")
    parser.add_argument(
        "--target-dtype",
        type=str,
        default="float32",
        help="Normalize embeddings to this dtype before comparing (e.g., float32)",
    )
    args = parser.parse_args()

    target_label = normalize_dtype_label(args.target_dtype)
    if target_label not in DTYPE_NAME_TO_NUMPY:
        supported = ", ".join(sorted(DTYPE_NAME_TO_NUMPY.keys()))
        raise ValueError(f"Unsupported target dtype '{args.target_dtype}'. Supported: {supported}")
    target_dtype = np.dtype(DTYPE_NAME_TO_NUMPY[target_label])

    emb_a, hist_a, dup_a = load_embeddings_by_query(args.file_a, target_dtype)
    emb_b, hist_b, dup_b = load_embeddings_by_query(args.file_b, target_dtype)

    queries_a = set(emb_a.keys())
    queries_b = set(emb_b.keys())

    common_queries = sorted(queries_a & queries_b)
    only_a = queries_a - queries_b
    only_b = queries_b - queries_a

    print(f"file_a: {args.file_a} has {len(queries_a)} unique queries")
    print(f"file_b: {args.file_b} has {len(queries_b)} unique queries")
    print(f"common queries: {len(common_queries)}")
    print(f"only in A: {len(only_a)}")
    print(f"only in B: {len(only_b)}")
    print(f"target dtype: {target_label}")
    print(f"file_a data_type histogram: {render_histogram(hist_a)}")
    print(f"file_b data_type histogram: {render_histogram(hist_b)}")

    if dup_a:
        print(f"Warning: skipped {dup_a} duplicate queries in file_a (keeping first occurrence)")
    if dup_b:
        print(f"Warning: skipped {dup_b} duplicate queries in file_b (keeping first occurrence)")

    if not common_queries:
        print("No overlapping queries, nothing to compare.")
        return

    cosines: List[float] = []
    l2s: List[float] = []
    per_query: List[Tuple[str, float, float]] = []

    for query in common_queries:
        vec_a = emb_a[query]
        vec_b = emb_b[query]
        cos, l2 = cosine_and_l2(vec_a, vec_b)
        cosines.append(cos)
        l2s.append(l2)
        per_query.append((query, cos, l2))

    cos_valid = [v for v in cosines if not math.isnan(v)]
    cos_mean, cos_min, cos_max = summarize(cos_valid)
    l2_mean, l2_min, l2_max = summarize(l2s)

    print("\nCosine similarity stats (non-NaN):")
    print(f"  mean: {cos_mean:.6f}, min: {cos_min:.6f}, max: {cos_max:.6f}")

    print("\nL2 distance stats:")
    print(f"  mean: {l2_mean:.6f}, min: {l2_min:.6f}, max: {l2_max:.6f}")

    top_k = max(0, args.top_k)
    if top_k > 0:
        per_query.sort(key=lambda item: item[2], reverse=True)
        print(f"\nTop-{top_k} queries by L2 distance:")
        for query, cos, l2 in per_query[:top_k]:
            print(f"  query={query!r}: L2={l2:.6f}, cos={cos:.6f}")


if __name__ == "__main__":
    main()
