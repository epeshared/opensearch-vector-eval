# ivf_index.py

from __future__ import annotations

from typing import Any, Dict, List

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError, ConflictError


# =========================
#  训练索引用的工具函数
# =========================

def recreate_train_index(
    client: OpenSearch,
    index_name: str,
    dimension: int,
    vector_field: str = "vector",
) -> None:
    """
    删除并重建用于训练 IVF 模型的索引。
    注意：在当前 OpenSearch 版本里，不支持 dense_vector，这里用 knn_vector。
    """
    # 删除旧索引
    if client.indices.exists(index=index_name):
        print(f"[TrainIndex] delete old index {index_name}")
        client.indices.delete(index=index_name)

    # 创建新索引：训练索引也用 knn_vector，方便 _train API 直接读取
    body = {
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    # 训练索引只用来存 raw 向量，不需要 method / model_id
                }
            }
        }
    }
    client.indices.create(index=index_name, body=body)
    print(f"[TrainIndex] created index={index_name}, dim={dimension}, field={vector_field}")


def ingest_train_vectors(
    client: OpenSearch,
    index_name: str,
    vectors: Any,
    vector_field: str = "vector",
) -> None:
    """
    将训练用的向量批量写入 Train 索引。
    vectors 通常是一个 (N, dim) 的 numpy 数组或类似结构。

    Parameters
    ----------
    client : OpenSearch
        已连接的 OpenSearch 客户端。
    index_name : str
        训练索引名，例如 "dbpedia-train"。
    vectors : Any
        一般为形如 (N, dim) 的 numpy.ndarray / torch.Tensor 等，需支持 .shape 和 .tolist()。
    vector_field : str, default "vector"
        存放训练向量的字段名，必须与 recreate_train_index 时的字段名一致。
    """
    print(f"[TrainIndex] ingest {vectors.shape[0]} training vectors to index={index_name}, field={vector_field}")

    actions: List[Dict[str, Any]] = []

    for i, vec in enumerate(vectors):
        # bulk 的每条 doc 由两行构成：action 行 + 源数据行
        actions.append({"index": {"_index": index_name, "_id": i}})
        actions.append({vector_field: vec.tolist()})

        # 简单分批写入，避免一次 body 过大
        if len(actions) >= 2000:  # 1000 条 doc * 2 行 bulk action
            client.bulk(body=actions)
            actions.clear()

    if actions:
        client.bulk(body=actions)

    # 确保后续训练模型时这些向量都可见
    client.indices.refresh(index=index_name)
    print("[TrainIndex] ingest done & index refreshed")


# =========================
#  FAISS IVF 模型训练
# =========================

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError, ConflictError


def train_faiss_ivf_model(
    client: OpenSearch,
    *,
    model_id: str,
    dimension: int,
    nlist: int,
    metric: str,
    train_index: str,
    train_field: str = "vector",
    recreate_model: bool = False,
    nprobes: int | None = 32,   # 可选的 nprobes
):
    """
    使用 k-NN 插件的 Train API 训练一个 FAISS IVF 模型。

    POST /_plugins/_knn/models/{model_id}/_train
    {
      "training_index": "...",
      "training_field": "...",
      "dimension": ...,
      "description": "...",
      "method": {
        "name": "ivf",
        "engine": "faiss",
        "space_type": "innerproduct" or "l2",
        "parameters": {
          "nlist": <int>,
          "nprobes": <int>   # (可选)
        }
      }
    }

    约定语义：
    - recreate_model=False 且模型已存在 => 复用旧模型，不报错。
    - recreate_model=True  且模型已存在 => 我们先尝试 DELETE，如果还是报 "already exists" 那就抛错。
    """

    # 1) 如需重建，则先删旧 model（忽略 404；如果还在 training，则给出清晰提示）
    if recreate_model:
        try:
            print(f"[Model] try delete existing model_id='{model_id}' before retraining")
            client.transport.perform_request(
                "DELETE",
                f"/_plugins/_knn/models/{model_id}",
            )
            print(f"[Model] deleted existing model_id='{model_id}'")
        except NotFoundError:
            print(f"[Model] model_id='{model_id}' does not exist, nothing to delete")
        except ConflictError as e:
            msg = str(getattr(e, "info", e))
            if "Model is still in training" in msg:
                raise RuntimeError(
                    f"[Model] cannot delete model_id='{model_id}' because it is still in training.\n"
                    f"- 说明上一次训练还没结束或者状态卡住了；\n"
                    f"- 可以临时换一个新的 model_id（例如加 _v2 后缀），\n"
                    f"  或者等 model 状态不是 training 后再删除。\n"
                ) from e
            raise

    # 2) 组装 IVF 参数：nlist 必须有，nprobes 可选
    params: dict = {
        "nlist": nlist,
    }
    if nprobes is not None:
        params["nprobes"] = nprobes

    body = {
        "training_index": train_index,
        "training_field": train_field,
        "dimension": dimension,
        "description": f"FAISS IVF model {model_id}",
        "method": {
            "name": "ivf",
            "engine": "faiss",
            "space_type": metric,  # "innerproduct" or "l2"
            "parameters": params,
        },
    }

    print(
        f"[ModelTrain] start training model_id={model_id} "
        f"with IVF-Flat (faiss, {metric}), nlist={nlist}, "
        f"nprobes={nprobes}, training_index={train_index}/{train_field}"
    )

    try:
        resp = client.transport.perform_request(
            "POST",
            f"/_plugins/_knn/models/{model_id}/_train",
            body=body,
        )
        print(f"[ModelTrain] response: {resp}")
        return resp
    except RequestError as e:
        msg = str(e.info) if hasattr(e, "info") else str(e)

        # 情况 1：model_id 已存在
        if 'already exists' in msg:
            if recreate_model:
                # 你明确说了要重训，但 API 又提示已存在，那就说明 delete/train 过程中有问题
                raise RuntimeError(
                    f"[ModelTrain] model_id='{model_id}' already exists even after trying to recreate.\n"
                    f"请检查：\n"
                    f"  - 是否有并行脚本在同时训练同一个 model_id；\n"
                    f"  - 或者手动 DELETE /_plugins/_knn/models/{model_id} 后再运行。\n"
                ) from e
            else:
                # 默认行为：模型已存在，直接复用，不报错
                print(
                    f"[ModelTrain] model_id='{model_id}' already exists, "
                    f"recreate_model=False, will reuse existing model."
                )
                return {"model_id": model_id, "status": "already_exists"}

        # 情况 2：老版本 API 写法错误（body 里带了 model_id）
        if "model_id" in msg and "not a valid parameter" in msg:
            raise RuntimeError(
                "[ModelTrain] Train API 调用格式不对（model_id 不应该出现在请求 body 的顶层）。\n"
                "当前代码已经按 OpenSearch 3.x 文档修正，\n"
                "如果仍然看到这个错误，请检查是否还有旧版本脚本在调用 "
                "`POST /_plugins/_knn/models/_train`。"
            ) from e

        # 其他错误照旧抛出
        raise


# =========================
#  IVF 索引创建
# =========================

def recreate_ivf_index(
    client: OpenSearch,
    index_name: str,
    *,
    dimension: int,
    model_id: str,
    metric: str,
    vector_field: str = "vector",
    nprobes: int = 32,
) -> None:
    """
    删除并重建使用 FAISS IVF-Flat 的 k-NN 索引。

    Parameters
    ----------
    client : OpenSearch
        已连接的 OpenSearch 客户端。
    index_name : str
        IVF 索引名，例如 "dbpedia-faiss-ivfflat-ip"。
    dimension : int
        向量维度，例如 1536。
    model_id : str
        已训练好的 k-NN 模型 ID，例如 "dbpedia-faiss-ivfflat-ip_model"。
    metric : str
        距离度量类型，常用 "innerproduct" 或 "l2"。
    vector_field : str, default "vector"
        主数据索引中用于搜索的向量字段名。
    nprobes : int, default 32
        搜索时探测的倒排桶数量（越大召回越好，但性能越低）。
    """
    if client.indices.exists(index=index_name):
        print(f"[IVFIndex] delete old index {index_name}")
        client.indices.delete(index=index_name)

    # k-NN 插件的 IVF-Flat 索引创建配置
    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.nprobes": nprobes,
            }
        },
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "ivf",
                        "space_type": metric,
                        "engine": "faiss",
                        # nlist 由模型决定，一般不用在这里重复配置
                        "parameters": {},
                        "model_id": model_id,
                    },
                }
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    print(
        f"[IVFIndex] created index={index_name}, dim={dimension}, "
        f"model_id={model_id}, metric={metric}, nprobes={nprobes}, field={vector_field}"
    )
