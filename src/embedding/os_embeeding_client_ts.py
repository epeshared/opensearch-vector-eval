# os/text_embedding_client.py

from typing import List, Optional, Tuple, Any, Dict
import requests


class OpenSearchTextEmbedder:
    """
    简单的 OpenSearch ML Commons text_embedding 客户端。

    使用方法：
        embedder = OpenSearchTextEmbedder(os_url, model_id)
        vecs = embedder.embed(["hello"])
    """

    def __init__(
        self,
        os_url: str,
        model_id: str,
        timeout: float = 30.0,
        auth: Optional[Tuple[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        self.os_url = os_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.auth = auth
        self.headers = headers or {}
        self.verify_ssl = verify_ssl

    def embed(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenSearch text_embedding 模型。"""
        url = f"{self.os_url}/_plugins/_ml/_predict/text_embedding/{self.model_id}"

        # 与 test_text_embedding.sh 保持一致：只设置 text_docs + return_number
        payload = {
            "text_docs": texts,
            "return_number": True,
        }

        resp = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
            headers=self.headers,
            auth=self.auth,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        data = resp.json()

        return self._parse_embedding(data, len(texts))

    @staticmethod
    def _parse_embedding(data: Dict[str, Any], expected_batch: int):
        """解析 OpenSearch text_embedding 响应结构，返回 sentence_embedding。"""

        try:
            results = data["inference_results"]
        except Exception as e:
            raise RuntimeError(f"Invalid response: {data}") from e

        if not isinstance(results, list) or not results:
            raise RuntimeError(f"Invalid response: {data}")

        vectors: List[List[float]] = []

        # 删除 target_response 后，TEXT_EMBEDDING 可能按 batch 返回，
        # 通常 inference_results[i].output[0] 即为该条的 embedding。
        for res in results:
            outputs = res.get("output") or []
            if not outputs:
                raise RuntimeError(f"Invalid response: {data}")
            out = outputs[0]
            shape = out.get("shape")
            flat = out.get("data")
            if not isinstance(shape, list) or not isinstance(flat, list):
                raise RuntimeError(f"Invalid response: {data}")

            if len(shape) == 1:
                # shape = [dim]
                vectors.append(flat)
            elif len(shape) == 2:
                # shape = [1, dim] 或 [batch, dim]
                rows, dim = shape
                if rows == 1:
                    vectors.append(flat[:dim])
                elif rows == expected_batch:
                    # 一次返回整个 batch
                    for i in range(rows):
                        start = i * dim
                        end = start + dim
                        vectors.append(flat[start:end])
                else:
                    raise RuntimeError(f"Unsupported batch shape: {shape}")
            else:
                raise RuntimeError(f"Unsupported output shape: {shape}")

        if expected_batch != len(vectors):
            raise RuntimeError(
                f"Batch mismatch: expected {expected_batch}, got {len(vectors)}"
            )

        return vectors
