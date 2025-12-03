#!/usr/bin/env bash
set -euo pipefail


WOKR_HOME="$PWD/../"
echo "WOKR_HOME: $WOKR_HOME"

# ====== 可配置参数 ======
OS_URL=${OS_URL:-"http://localhost:9200"}
INPUT=${INPUT:-"/home/xtang/datasets/yahoo_answers_title_answer.jsonl"}
OUTPUT=${OUTPUT:-"output/embeddings/yahoo_vecs.jsonl"}
BATCH_SIZE=${BATCH_SIZE:-10}
MAX_QUERIES=${MAX_QUERIES:-"10000"}

echo "Using OpenSearch at: $OS_URL"

# ====== 1. 自动从 OpenSearch 找一个 TEXT_EMBEDDING 的已部署模型 ======
echo "Looking for deployed TEXT_EMBEDDING model ..."

MODEL_ID=$(
  curl -s -XPOST "$OS_URL/_plugins/_ml/models/_search" \
    -H 'Content-Type: application/json' \
    -d '{
      "query": {
        "bool": {
          "must": [
            { "term": { "algorithm": "TEXT_EMBEDDING" }},
            { "term": { "model_state": "DEPLOYED" }}
          ]
        }
      },
      "size": 1
    }' | jq -r '.hits.hits[0]._id'
)

if [ -z "$MODEL_ID" ] || [ "$MODEL_ID" = "null" ]; then
  echo "ERROR: 找不到已部署的 TEXT_EMBEDDING 模型，请先确认模型已经 register + deploy."
  exit 1
fi

echo "Found model_id: $MODEL_ID"

# ====== 2. 调用你的 Python 脚本做 embedding ======
echo "Running Yahoo embedding job ..."

PY_ARGS=(
  "$WOKR_HOME/src/yahoo_qeury_embedding.py"
  --input "$INPUT"
  --output "$OUTPUT"
  --os-url "$OS_URL"
  --model-id "$MODEL_ID"
  --batch-size "$BATCH_SIZE"
)

if [ -n "$MAX_QUERIES" ]; then
  echo "Limiting to MAX_QUERIES=$MAX_QUERIES"
  PY_ARGS+=(--max-queries "$MAX_QUERIES")
fi

python "${PY_ARGS[@]}"

echo "Done. Embeddings written to: $OUTPUT"
