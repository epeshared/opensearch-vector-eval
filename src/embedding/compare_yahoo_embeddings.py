import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import math


def load_embeddings(path: Path) -> Dict[str, List[float]]:
    """Load id -> embedding vector from yahoo_vecs.jsonl-like file."""
    mapping: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj.get("id"))
            emb = obj.get("embedding")
            if _id is None or emb is None:
                continue
            mapping[_id] = emb
    return mapping


def cosine_and_l2(a: List[float], b: List[float]) -> Tuple[float, float]:
    if len(a) != len(b):
        raise ValueError(f"Vector dim mismatch: {len(a)} vs {len(b)}")

    dot = 0.0
    na = 0.0
    nb = 0.0
    l2 = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
        diff = x - y
        l2 += diff * diff

    if na == 0.0 or nb == 0.0:
        cos = float("nan")
    else:
        cos = dot / math.sqrt(na * nb)
    return cos, math.sqrt(l2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two yahoo_vecs.jsonl embedding files.")
    parser.add_argument("--file-a", type=Path, required=True, help="Baseline embeddings JSONL (e.g. original model)")
    parser.add_argument("--file-b", type=Path, required=True, help="New embeddings JSONL (e.g. exported TS model)")
    parser.add_argument("--top-k", type=int, default=10, help="Show top-K largest L2 diffs")
    args = parser.parse_args()

    emb_a = load_embeddings(args.file_a)
    emb_b = load_embeddings(args.file_b)

    ids_a = set(emb_a.keys())
    ids_b = set(emb_b.keys())

    common_ids = sorted(ids_a & ids_b)
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"file_a: {args.file_a} has {len(ids_a)} ids")
    print(f"file_b: {args.file_b} has {len(ids_b)} ids")
    print(f"common ids: {len(common_ids)}")
    print(f"only in A: {len(only_a)}")
    print(f"only in B: {len(only_b)}")

    if not common_ids:
        print("No overlapping ids, nothing to compare.")
        return

    cosines: List[float] = []
    l2s: List[float] = []
    per_id: List[Tuple[str, float, float]] = []  # (id, cos, l2)

    for _id in common_ids:
        a = emb_a[_id]
        b = emb_b[_id]
        cos, l2 = cosine_and_l2(a, b)
        cosines.append(cos)
        l2s.append(l2)
        per_id.append((_id, cos, l2))

    def summarize(values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan")
        v_min = min(values)
        v_max = max(values)
        v_mean = sum(values) / len(values)
        return v_mean, v_min, v_max

    cos_mean, cos_min, cos_max = summarize([v for v in cosines if not math.isnan(v)])
    l2_mean, l2_min, l2_max = summarize(l2s)

    print("\nCosine similarity stats (over non-NaN):")
    print(f"  mean: {cos_mean:.6f}, min: {cos_min:.6f}, max: {cos_max:.6f}")

    print("\nL2 distance stats:")
    print(f"  mean: {l2_mean:.6f}, min: {l2_min:.6f}, max: {l2_max:.6f}")

    # Show top-K largest L2 diffs
    k = max(0, args.top_k)
    if k > 0:
        per_id_sorted = sorted(per_id, key=lambda x: x[2], reverse=True)
        print(f"\nTop-{k} examples by L2 distance:")
        for _id, cos, l2 in per_id_sorted[:k]:
            print(f"  id={_id}: L2={l2:.6f}, cos={cos:.6f}")


if __name__ == "__main__":
    main()
