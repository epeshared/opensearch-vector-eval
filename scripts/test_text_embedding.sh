#!/usr/bin/env bash
set -euo pipefail

# Simple TEXT_EMBEDDING prediction tester

OS_HOST="${OS_HOST:-http://localhost:9200}"
MODEL_ID="${MODEL_ID:-}"
POLL_INTERVAL=${POLL_INTERVAL:-5}
OUTPUT="${OUTPUT:-output/embeddings/test_prediction.json}"

OS_USER="${OS_USER:-}"
OS_PASS="${OS_PASS:-}"

echo "MODEL_ID: $MODEL_ID"

curl_os() {
  if [[ -n "$OS_USER" && -n "$OS_PASS" ]]; then
    curl -sS -u "${OS_USER}:${OS_PASS}" "$@"
  else
    curl -sS "$@"
  fi
}

# 如果没有显式提供 MODEL_ID，则自动从 OpenSearch 中找一个已部署的 TEXT_EMBEDDING 模型
if [[ -z "$MODEL_ID" ]]; then
  echo "Looking for deployed TEXT_EMBEDDING model ..."
  MODEL_ID=$(curl_os -XPOST "${OS_HOST}/_plugins/_ml/models/_search" \
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
    }' | jq -r '.hits.hits[0]._id')

  if [[ -z "$MODEL_ID" || "$MODEL_ID" == "null" ]]; then
    echo "ERROR: 没有找到已部署的 TEXT_EMBEDDING 模型，请先运行 setup_embedding.sh 注册并部署模型。"
    exit 1
  fi
fi

echo "Using model_id: $MODEL_ID"

TEXTS=(
  "OpenSearch is a distributed search and analytics engine."
  "Embedding models map text into dense vectors."
)

# payload=$(jq -n --arg t1 "${TEXTS[0]}" --arg t2 "${TEXTS[1]}" '{text_docs: [$t1, $t2], return_number: true, target_response: ["sentence_embedding"]}')

payload=$(jq -n --arg t1 "${TEXTS[0]}" --arg t2 "${TEXTS[1]}" '{text_docs: [$t1, $t2], return_number: true}')


mkdir -p "$(dirname "$OUTPUT")"

response=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/_predict/text_embedding/${MODEL_ID}" \
  -H 'Content-Type: application/json' \
  -d "$payload")

printf '%s
' "$response" | tee "$OUTPUT" | jq .

echo "Saved response to $OUTPUT"
