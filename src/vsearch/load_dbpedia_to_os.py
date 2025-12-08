#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 dbpedia-openai-1000k-angular.hdf5 读取向量，
通过 IVF-Flat (Faiss, innerproduct) 存到 OpenSearch，并做一次向量检索。

模型 ID 不再通过命令行传入，而是自动生成：
    model_id = <ivf_index> + "_model"

逻辑说明：
- --recreate-train-index: 删除并重建训练索引（若不存在则自动创建）；
- --recreate-ivf-index:  删除并重建 IVF 索引，同时会删除并重训同名 model。
"""

from __future__ import annotations

import argparse
from typing import Tuple
from urllib.parse import urlparse

import h5py
import numpy as np
from opensearchpy import OpenSearch

from ivf_index import (
    recreate_train_index,
    ingest_train_vectors,
    train_faiss_ivf_model,
    recreate_ivf_index,
)

# 统一的向量字段名
TRAIN_VECTOR_FIELD = "vector"
IVF_VECTOR_FIELD = "vector"


# ========== 1. 读 HDF5 数据 ==========

def load_ann_hdf5(
    path: str,
    max_train: int | None = None,
    max_test: int | None = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 ann-benchmarks 风格的 HDF5 文件:
    - /train: 向量库
    - /test:  查询向量
    """
    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])

    if max_train is not None:
        train = train[:max_train]
    if max_test is not None:
        test = test[:max_test]

    print(f"[HDF5] train shape={train.shape}, test shape={test.shape}")
    return train.astype(np.float32), test.astype(np.float32)


# ========== 2. OpenSearch 客户端 ==========

def create_os_client(
    host: str = "localhost",
    port: int = 9200,
    user: str | None = "admin",
    password: str | None = "admin",
    use_ssl: bool = False,
    verify_certs: bool = False,
) -> OpenSearch:
    http_auth = (user, password) if user and password else None

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=http_auth,
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_show_warn=False,
        timeout=120,
    )
    info = client.info()
    print(
        f"[OpenSearch] connected, cluster={info.get('cluster_name')}, "
        f"version={info.get('version', {}).get('number')}"
    )
    return client


# ========== 3. 通用 knn 查询 ==========

def knn_search(
    client: OpenSearch,
    index_name: str,
    query_vector: np.ndarray,
    k: int = 10,
    vector_field: str = IVF_VECTOR_FIELD,
) -> None:
    """
    用 k-NN Query 做向量检索。
    space_type=innerproduct 已在模型里指定。
    这里使用老语法：
        "knn": { "<field>": { "vector": [...], "k": k } }
    """
    body = {
        "size": k,
        "query": {
            "knn": {
                vector_field: {
                    "vector": query_vector.tolist(),
                    "k": k,
                }
            }
        }
    }

    resp = client.search(index=index_name, body=body)
    hits = resp["hits"]["hits"]
    print(f"[Search] top-{k} results:")
    for h in hits:
        _id = h["_id"]
        score = h["_score"]
        print(f"  id={_id}, score={score}")


# ========== 4. main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=str,
        required=True,
        help="Path to dbpedia-openai-1000k-angular.hdf5",
    )
    parser.add_argument("--os-url", type=str, default="http://localhost:9200")
    parser.add_argument("--user", type=str, default="admin")
    parser.add_argument("--password", type=str, default="admin")

    parser.add_argument(
        "--train-index",
        type=str,
        default="dbpedia-train",
        help="Index used to store training vectors for IVF.",
    )
    parser.add_argument(
        "--ivf-index",
        type=str,
        default="dbpedia-faiss-ivfflat-ip",
        help="Main IVF-Flat index name storing all vectors.",
    )

    parser.add_argument(
        "--max-train",
        type=int,
        default=100000,
        help="How many vectors to use for IVF training (subset of train).",
    )
    parser.add_argument("--max-test", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobes", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)

    # 是否删除旧 index（train / ivf）
    parser.add_argument(
        "--recreate-train-index",
        action="store_true",
        help="If set, delete and recreate train index if it exists.",
    )
    parser.add_argument(
        "--recreate-ivf-index",
        action="store_true",
        help=(
            "If set, delete and recreate IVF index if it exists, "
            "and also delete & retrain the k-NN model with the same model_id."
        ),
    )

    args = parser.parse_args()

    parsed = urlparse(args.os_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9200
    use_ssl = parsed.scheme == "https"

    # Step 1: 读 HDF5
    train_vecs, test_vecs = load_ann_hdf5(
        args.hdf5, max_train=args.max_train, max_test=args.max_test
    )
    dim = train_vecs.shape[1]

    # Step 2: OS client
    client = create_os_client(
        host=host,
        port=port,
        user=args.user,
        password=args.password,
        use_ssl=use_ssl,
        verify_certs=False,
    )

    # 自动生成 model_id（对用户透明）：<ivf_index> + "_model"
    model_id = f"{args.ivf_index}_model"
    print(f"[Model] use model_id='{model_id}' for IVF index '{args.ivf_index}'")

    # ========= Step 3: 训练 IVF 模型 =========

    # 3.1 训练索引：根据 --recreate-train-index 决定是否强制重建
    if args.recreate_train_index:
        recreate_train_index(
            client=client,
            index_name=args.train_index,
            dimension=dim,
            vector_field=TRAIN_VECTOR_FIELD,
        )
    else:
        # 不指定重建时，如果索引不存在则创建
        if not client.indices.exists(index=args.train_index):
            recreate_train_index(
                client=client,
                index_name=args.train_index,
                dimension=dim,
                vector_field=TRAIN_VECTOR_FIELD,
            )

    # 3.2 写入训练向量（load_ann_hdf5 已经过 max_train 截断）
    ingest_train_vectors(
        client=client,
        index_name=args.train_index,
        vectors=train_vecs,
        vector_field=TRAIN_VECTOR_FIELD,
    )

    # 3.3 训练 FAISS IVF-Flat 模型
    train_faiss_ivf_model(
        client=client,
        model_id=model_id,
        dimension=dim,
        nlist=args.nlist,
        metric="innerproduct",
        train_index=args.train_index,
        train_field=TRAIN_VECTOR_FIELD,
        # 若指定重建 IVF index，则 model 也一并重建
        recreate_model=args.recreate_ivf_index,
        nprobes=args.nprobes,
    )

    # ========= Step 4: 创建 IVF index + 导入向量 =========

    # 根据 --recreate-ivf-index 决定是否删除旧 IVF index
    if args.recreate_ivf_index:
        recreate_ivf_index(
            client=client,
            index_name=args.ivf_index,
            dimension=dim,
            model_id=model_id,
            metric="innerproduct",
            vector_field=IVF_VECTOR_FIELD,
            nprobes=args.nprobes,
        )
    else:
        if not client.indices.exists(index=args.ivf_index):
            recreate_ivf_index(
                client=client,
                index_name=args.ivf_index,
                dimension=dim,
                model_id=model_id,
                metric="innerproduct",
                vector_field=IVF_VECTOR_FIELD,
                nprobes=args.nprobes,
            )

    # 把向量写入 IVF 索引（这里简单用同一批 train_vecs，当作完整向量库）
    ingest_train_vectors(
        client=client,
        index_name=args.ivf_index,
        vectors=train_vecs,
        vector_field=IVF_VECTOR_FIELD,   # 要和 recreate_ivf_index 里的 vector_field 保持一致
    )

    # ========= Step 5: 做一次检索 =========
    if test_vecs.shape[0] > 0:
        query_vec = test_vecs[0]
        knn_search(
            client=client,
            index_name=args.ivf_index,
            query_vector=query_vec,
            k=args.k,
            vector_field=IVF_VECTOR_FIELD,
        )
    else:
        print("[Search] no test vectors found, skip search")


if __name__ == "__main__":
    main()
