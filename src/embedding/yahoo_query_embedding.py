# yahoo_to_embeddings.py

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Any, Dict

from os_embeeding_client import OpenSearchTextEmbedder


def read_queries(jsonl_path: Path, query_field="query", fallback_field="title", id_field="id"):
    """读取 JSONL，每行提供 (id, text)。兼容 dict 或 list 结构。"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            text = None
            item_id: Any = lineno  # fallback id to keep determinism

            if isinstance(obj, dict):
                text = obj.get(query_field) or obj.get(fallback_field)
                item_id = obj.get(id_field, lineno)
            elif isinstance(obj, list):
                # Yahoo Answers 数据是 [question, answer]，优先取 list 中第一个非空字符串
                for entry in obj:
                    if isinstance(entry, str) and entry.strip():
                        text = entry.strip()
                        break
            else:
                continue

            if not text:
                continue

            yield item_id, text


def batched(iterable, batch_size: int):
    """简单 batch 工具。"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def take_first(iterable, limit: int):
    """限制迭代器只返回前 limit 个元素。"""
    if limit is None or limit < 0:
        yield from iterable
        return

    for idx, item in enumerate(iterable):
        if idx >= limit:
            break
        yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--os-url", type=str, default="http://localhost:9200")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--username", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="只转换前 N 条 query，默认全部处理")
    args = parser.parse_args()

    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)

    embedder = OpenSearchTextEmbedder(
        os_url=args.os_url,
        model_id=args.model_id,
        auth=auth,
        verify_ssl=not args.insecure,
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    embed_time_total = 0.0
    query_iter = read_queries(args.input)
    query_iter = take_first(query_iter, args.max_queries)

    # 为了显示进度条，先把要处理的 query 收集到列表中
    all_items = list(query_iter)
    total_planned = len(all_items)
    if total_planned == 0:
        print("No queries to embed.")
        return

    def print_progress(done: int):
        ratio = done / total_planned
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {done}/{total_planned} ({ratio*100:.1f}%)", end="", flush=True)

    with output.open("w", encoding="utf-8") as fout:
        for batch in batched(all_items, args.batch_size):
            ids, texts = zip(*batch)
            start = time.time()
            vectors, data_types = embedder.embed(list(texts), return_info=True)
            embed_time_total += time.time() - start
            for _id, q, v, dtype in zip(ids, texts, vectors, data_types):
                fout.write(json.dumps({
                    "id": _id,
                    "query": q,
                    "embedding": v,
                    "data_type": dtype,
                }) + "\n")
                total += 1
                print_progress(total)

    print()  # 换行
    print(f"Done. Wrote {total} embeddings  {output}")
    if embed_time_total > 0 and total > 0:
        qps = total / embed_time_total
        print(f"Total embed time: {embed_time_total:.3f}s, QPS: {qps:.2f}")
    else:
        print("No embedding time recorded (no queries?)")


if __name__ == "__main__":
    main()
