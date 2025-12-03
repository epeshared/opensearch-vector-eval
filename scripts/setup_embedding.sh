#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"

########################################
# 配置区：按需修改
########################################

# OpenSearch 地址
OS_HOST="${OS_HOST:-http://localhost:9200}"

# 如果需要 basic auth，就在这里配；不需要就留空
OS_USER="${OS_USER:-}"
OS_PASS="${OS_PASS:-}"

# 选一个 OpenSearch 官方提供的 TEXT_EMBEDDING 模型名称
# 这个名字需要在你当前 OpenSearch 版本的 "pretrained models" 列表里存在
MODEL_NAME="${MODEL_NAME:-huggingface/sentence-transformers/msmarco-distilbert-base-tas-b}"
MODEL_VERSION="${MODEL_VERSION:-1.0.1}"
MODEL_FORMAT="${MODEL_FORMAT:-TORCH_SCRIPT}"

# 轮询任务的时间间隔（秒）
POLL_INTERVAL=10

########################################
# 工具函数
########################################

curl_os() {
  # 自动根据是否配置了用户名密码来加 -u
  if [[ -n "$OS_USER" && -n "$OS_PASS" ]]; then
    curl -sS -u "${OS_USER}:${OS_PASS}" "$@"
  else
    curl -sS "$@"
  fi
}

echo_title() {
  echo
  echo "========================================"
  echo "$@"
  echo "========================================"
}

########################################
# 1. 检查集群健康 & ML 插件
########################################

echo_title "1. 检查集群健康状态"
curl_os "${OS_HOST}/_cluster/health?pretty"

echo_title "2. 检查 ML Commons 插件是否存在"
curl_os "${OS_HOST}/_cat/plugins?v" | grep -E "ml|opensearch-ml" || {
  echo "ERROR: 没有看到 ML 插件 (ml-commons / opensearch-ml)，请确认插件已安装并启用。"
  exit 1
}

########################################
# 2. 设置 ML 相关集群参数
########################################

echo_title "3. 配置 ML Commons 相关 cluster settings"

curl_os -X PUT "${OS_HOST}/_cluster/settings" \
  -H 'Content-Type: application/json' -d '{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "false",
    "plugins.ml_commons.native_memory_threshold": "99",
    "plugins.ml_commons.allow_registering_model_via_url": "true"
  }
}' | jq .

########################################
# 3. 注册预训练 TEXT_EMBEDDING 模型
########################################

echo_title "4. 注册预训练 embedding 模型: ${MODEL_NAME}"

REGISTER_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/models/_register" \
  -H 'Content-Type: application/json' -d "{
  \"name\": \"${MODEL_NAME}\",
  \"version\": \"${MODEL_VERSION}\",
  \"model_format\": \"${MODEL_FORMAT}\"
}")

echo "${REGISTER_RESP}" | jq .

TASK_ID=$(echo "${REGISTER_RESP}" | jq -r '.task_id // empty')

if [[ -z "${TASK_ID}" || "${TASK_ID}" == "null" ]]; then
  echo "ERROR: 注册模型时没有拿到 task_id，返回内容如下："
  echo "${REGISTER_RESP}"
  exit 1
fi

echo "注册任务 task_id: ${TASK_ID}"

########################################
# 4. 轮询注册任务直到完成，拿到 model_id
########################################

echo_title "5. 等待模型注册完成 (task_id=${TASK_ID})"

MODEL_ID=""
while true; do
  TASK_RESP=$(curl_os "${OS_HOST}/_plugins/_ml/tasks/${TASK_ID}")
  echo "${TASK_RESP}" | jq .

  STATE=$(echo "${TASK_RESP}" | jq -r '.state // empty')
  if [[ "${STATE}" == "COMPLETED" ]]; then
    MODEL_ID=$(echo "${TASK_RESP}" | jq -r '.model_id // empty')
    break
  elif [[ "${STATE}" == "FAILED" || "${STATE}" == "ERROR" ]]; then
    echo "ERROR: 模型注册失败，任务详情："
    echo "${TASK_RESP}"
    exit 1
  else
    echo "当前任务状态: ${STATE:-unknown}，${POLL_INTERVAL}s 后重试..."
    sleep "${POLL_INTERVAL}"
  fi
done

if [[ -z "${MODEL_ID}" || "${MODEL_ID}" == "null" ]]; then
  echo "ERROR: 任务完成但没有拿到 model_id，任务详情："
  echo "${TASK_RESP}"
  exit 1
fi

echo "模型注册完成，model_id = ${MODEL_ID}"

########################################
# 5. 部署模型
########################################

echo_title "6. 部署模型 (model_id=${MODEL_ID})"

DEPLOY_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/models/${MODEL_ID}/_deploy")
echo "${DEPLOY_RESP}" | jq .

DEPLOY_TASK_ID=$(echo "${DEPLOY_RESP}" | jq -r '.task_id // empty')

if [[ -z "${DEPLOY_TASK_ID}" || "${DEPLOY_TASK_ID}" == "null" ]]; then
  echo "ERROR: 部署模型时没有拿到 task_id，返回内容如下："
  echo "${DEPLOY_RESP}"
  exit 1
fi

echo "部署任务 task_id: ${DEPLOY_TASK_ID}"

########################################
# 6. 轮询部署任务直到 COMPLETED
########################################

echo_title "7. 等待模型部署完成 (task_id=${DEPLOY_TASK_ID})"

while true; do
  DEPLOY_TASK_RESP=$(curl_os "${OS_HOST}/_plugins/_ml/tasks/${DEPLOY_TASK_ID}")
  echo "${DEPLOY_TASK_RESP}" | jq .

  STATE=$(echo "${DEPLOY_TASK_RESP}" | jq -r '.state // empty')
  if [[ "${STATE}" == "COMPLETED" ]]; then
    echo "模型部署完成。"
    break
  elif [[ "${STATE}" == "FAILED" || "${STATE}" == "ERROR" ]]; then
    echo "ERROR: 模型部署失败，任务详情："
    echo "${DEPLOY_TASK_RESP}"
    exit 1
  else
    echo "当前部署状态: ${STATE:-unknown}，${POLL_INTERVAL}s 后重试..."
    sleep "${POLL_INTERVAL}"
  fi
done

########################################
# 7. 用 TEXT_EMBEDDING 接口做一次预测
########################################

echo_title "8. 用 text_embedding 接口做一次 embedding 预测"

READABLE_TEXT_1="OpenSearch is a distributed search and analytics engine."
READABLE_TEXT_2="Embedding models map text into dense vectors."

PREDICT_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/_predict/text_embedding/${MODEL_ID}" \
  -H 'Content-Type: application/json' -d "{
  \"text_docs\": [
    \"${READABLE_TEXT_1}\",
    \"${READABLE_TEXT_2}\"
  ],
  \"return_number\": true,
  \"target_response\": [\"sentence_embedding\"]
}")

echo "${PREDICT_RESP}" | jq .

echo_title "脚本执行完毕：模型已注册、部署，并完成了一次 embedding 调用。"
