#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

########################################
# 配置区：按需修改
########################################

OS_HOST="${OS_HOST:-http://localhost:9200}"
OS_USER="${OS_USER:-}"
OS_PASS="${OS_PASS:-}"

# HF 模型 ID，用来下载 tokenizer
HF_MODEL_ID="${HF_MODEL_ID:-sentence-transformers/msmarco-distilbert-base-tas-b}"

# 本地 TorchScript 路径（已经由你 export_export_embedding_model.sh 生成）
LOCAL_TS_PATH="${LOCAL_TS_PATH:-/mnt/nvme2n1p1/xtang/opensearch-playground/opensearch-vector-eval/src/embedding/ts_models/sentence-transformers/msmarco_distilbert_base_ts.pt}"

# zip 输出路径
MODEL_ZIP_PATH="${MODEL_ZIP_PATH:-/mnt/nvme2n1p1/xtang/opensearch-playground/opensearch-vector-eval/src/embedding/ts_models/sentence-transformers/msmarco_distilbert_base_ts.zip}"

# model group 名
MODEL_GROUP_NAME="${MODEL_GROUP_NAME:-local_text_embedding_group}"

# 模型注册信息
MODEL_NAME="${MODEL_NAME:-sentence-transformers/msmarco-distilbert-base-tas-b}"
MODEL_VERSION="${MODEL_VERSION:-1.0.1-local}"
MODEL_DESCRIPTION="${MODEL_DESCRIPTION:-Local TorchScript text embedding model from sentence-transformers/msmarco-distilbert-base-tas-b}"
FUNCTION_NAME="${FUNCTION_NAME:-TEXT_EMBEDDING}"
MODEL_FORMAT="${MODEL_FORMAT:-TORCH_SCRIPT}"

# model_config 信息（按你这个 DistilBERT 模型来写）
MODEL_TYPE="${MODEL_TYPE:-distilbert}"
EMBEDDING_DIM="${EMBEDDING_DIM:-768}"
FRAMEWORK_TYPE="${FRAMEWORK_TYPE:-sentence_transformers}"

POLL_INTERVAL=10
HTTP_SERVER_PORT="${HTTP_SERVER_PORT:-8000}"

########################################
# 小工具函数
########################################

curl_os() {
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
# 0. 打包模型为 zip + 计算 size / sha256
########################################

echo_title "0. 打包 TorchScript 模型为 zip，并计算 size / sha256"

if [[ ! -f "$LOCAL_TS_PATH" ]]; then
  echo "ERROR: LOCAL_TS_PATH 不存在: $LOCAL_TS_PATH"
  exit 1
fi

PY_INFO=$(
  HF_MODEL_ID="$HF_MODEL_ID" \
  LOCAL_TS_PATH="$LOCAL_TS_PATH" \
  MODEL_ZIP_PATH="$MODEL_ZIP_PATH" \
  python - << 'PY'
import os, json, hashlib, zipfile
from transformers import AutoTokenizer, AutoConfig

hf_id = os.environ["HF_MODEL_ID"]
ts_path = os.environ["LOCAL_TS_PATH"]
zip_path = os.environ["MODEL_ZIP_PATH"]

ts_dir, ts_name = os.path.split(ts_path)

# 1) 下载 tokenizer（如果还没有）
tok_dir = os.path.join(ts_dir, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)

marker = os.path.join(tok_dir, ".downloaded")
if not os.path.exists(marker):
  print(f"[Info] 下载 tokenizer: {hf_id}")
  tok = AutoTokenizer.from_pretrained(hf_id)
  tok.save_pretrained(tok_dir)
  with open(marker, "w") as f:
    f.write("ok\n")

# 1.5) 读取 HF config，用于 TEXT_EMBEDDING all_config
print("[Info] 加载 HF config 用于 all_config")
cfg = AutoConfig.from_pretrained(hf_id)
all_config_json = cfg.to_json_string()

# 2) 打 zip：根目录只放 .pt + tokenizer 的文件
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
  # 模型本体
  zf.write(ts_path, arcname=os.path.basename(ts_path))
  # tokenizer 文件全部扁平放在根目录
  for fname in os.listdir(tok_dir):
    fpath = os.path.join(tok_dir, fname)
    if os.path.isfile(fpath):
      zf.write(fpath, arcname=fname)

# 3) 统计 size + sha256
size = os.path.getsize(zip_path)
h = hashlib.sha256()
with open(zip_path, "rb") as f:
  for chunk in iter(lambda: f.read(1 << 20), b""):
    h.update(chunk)

print(json.dumps({"size": size, "sha256": h.hexdigest(), "all_config": all_config_json}))
PY
)

MODEL_SIZE_BYTES=$(echo "$PY_INFO" | tail -n 1 | jq -r '.size')
MODEL_SHA256=$(echo "$PY_INFO"   | tail -n 1 | jq -r '.sha256')
MODEL_ALL_CONFIG=$(echo "$PY_INFO" | tail -n 1 | jq -r '.all_config')

echo "[Info] MODEL_ZIP_PATH      = $MODEL_ZIP_PATH"
echo "[Info] MODEL_SIZE_BYTES    = $MODEL_SIZE_BYTES"
echo "[Info] MODEL_SHA256        = $MODEL_SHA256"

# 启动本地 HTTP server，用于 URL 下载 zip
ZIP_DIR="$(dirname "$MODEL_ZIP_PATH")"
ZIP_BASENAME="$(basename "$MODEL_ZIP_PATH")"

echo "[Info] 在本地启动 HTTP server: dir=$ZIP_DIR, port=$HTTP_SERVER_PORT"
(
  cd "$ZIP_DIR"
  python -m http.server "$HTTP_SERVER_PORT" >/dev/null 2>&1
) &
HTTP_SERVER_PID=$!
sleep 1  # 给 server 一点启动时间

MODEL_URL="http://127.0.0.1:${HTTP_SERVER_PORT}/${ZIP_BASENAME}"
echo "[Info] 模型 URL = $MODEL_URL"

########################################
# 1. 检查集群健康
########################################

echo_title "1. 检查集群健康状态"
curl_os "${OS_HOST}/_cluster/health?pretty" || true

########################################
# 2. 检查 ML 插件
########################################

echo_title "2. 检查 ML Commons / opensearch-ml 插件是否存在"
curl_os "${OS_HOST}/_cat/plugins?v" | grep -E "ml|opensearch-ml" || {
  echo "ERROR: 没有发现 ML 插件 (ml-commons / opensearch-ml)，请确认插件已安装。"
  kill "$HTTP_SERVER_PID" || true
  exit 1
}

########################################
# 3. 调整 cluster settings
########################################

echo_title "3. 配置 ML Commons 相关 cluster settings（允许通过 URL 注册模型）"
curl_os -X PUT "${OS_HOST}/_cluster/settings" \
  -H 'Content-Type: application/json' -d '{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99",
    "plugins.ml_commons.allow_registering_model_via_url": "true"
  }
}' || true

########################################
# 4. 注册 / 获取 model group
########################################

echo_title "4. 注册 / 获取 model group: ${MODEL_GROUP_NAME}"

# 先查是否已存在同名 model group
SEARCH_BODY=$(jq -n --arg name "$MODEL_GROUP_NAME" '{
  "query": { "term": { "name": $name } }
}')
SEARCH_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/model_groups/_search" \
  -H 'Content-Type: application/json' -d "$SEARCH_BODY" || true)

EXISTING_GROUP_ID=$(echo "$SEARCH_RESP" | jq -r '.hits.hits[0]._id // empty' 2>/dev/null || true)

if [[ -n "$EXISTING_GROUP_ID" && "$EXISTING_GROUP_ID" != "null" ]]; then
  MODEL_GROUP_ID="$EXISTING_GROUP_ID"
  echo "[Info] 已存在同名 model group: ${MODEL_GROUP_NAME}, id=${MODEL_GROUP_ID}"
else
  echo "[Info] 未找到同名 model group，创建新的: ${MODEL_GROUP_NAME}"
  CREATE_BODY=$(jq -n --arg name "$MODEL_GROUP_NAME" '{ name: $name }')

  CREATE_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/model_groups/_register" \
    -H 'Content-Type: application/json' -d "$CREATE_BODY")

  echo "$CREATE_RESP" | jq . || true

  MODEL_GROUP_ID=$(echo "$CREATE_RESP" | jq -r '.model_group_id // empty')
  if [[ -z "$MODEL_GROUP_ID" || "$MODEL_GROUP_ID" == "null" ]]; then
    echo "ERROR: model group 注册失败"
    echo "$CREATE_RESP"
    kill "$HTTP_SERVER_PID" || true
    exit 1
  fi
fi

echo "[Info] model_group_id = ${MODEL_GROUP_ID}"

########################################
# 5. 注册本地 TorchScript 模型
########################################

echo_title "5. 注册本地 TorchScript 模型到该 model group"

REGISTER_JSON=$(jq -n \
  --arg name        "$MODEL_NAME" \
  --arg version     "$MODEL_VERSION" \
  --arg mgid        "$MODEL_GROUP_ID" \
  --arg desc        "$MODEL_DESCRIPTION" \
  --arg fn          "$FUNCTION_NAME" \
  --arg fmt         "$MODEL_FORMAT" \
  --arg hash        "$MODEL_SHA256" \
  --arg mtype       "$MODEL_TYPE" \
  --arg fw          "$FRAMEWORK_TYPE" \
  --arg url         "$MODEL_URL" \
  --arg allcfg      "$MODEL_ALL_CONFIG" \
  --argjson size    "$MODEL_SIZE_BYTES" \
  --argjson edim    "$EMBEDDING_DIM" '
  {
    name: $name,
    version: $version,
    model_group_id: $mgid,
    description: $desc,
    function_name: $fn,
    model_format: $fmt,
    model_content_size_in_bytes: $size,
    model_content_hash_value: $hash,
    model_config: {
      model_type: $mtype,
      embedding_dimension: $edim,
      framework_type: $fw,
      all_config: $allcfg
    },
    url: $url
  }')

echo "[Debug] Register model 请求体："
echo "$REGISTER_JSON" | jq . || echo "$REGISTER_JSON"

REGISTER_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/models/_register" \
  -H 'Content-Type: application/json' \
  -d "$REGISTER_JSON")

echo "$REGISTER_RESP" | jq . || true

TASK_ID=$(echo "$REGISTER_RESP" | jq -r '.task_id // empty')
if [[ -z "$TASK_ID" || "$TASK_ID" == "null" ]]; then
  echo "ERROR: 注册模型时没有拿到 task_id，返回内容如下："
  echo "$REGISTER_RESP"
  kill "$HTTP_SERVER_PID" || true
  exit 1
fi

echo "[Info] 注册任务 task_id = ${TASK_ID}"

########################################
# 6. 轮询注册任务直到 COMPLETED，拿到 model_id
########################################

echo_title "6. 等待模型注册完成 (task_id=${TASK_ID})"

MODEL_ID=""
while true; do
  TASK_RESP=$(curl_os "${OS_HOST}/_plugins/_ml/tasks/${TASK_ID}")
  echo "$TASK_RESP" | jq . || true

  STATE=$(echo "$TASK_RESP" | jq -r '.state // empty')
  if [[ "$STATE" == "COMPLETED" ]]; then
    MODEL_ID=$(echo "$TASK_RESP" | jq -r '.model_id // empty')
    break
  elif [[ "$STATE" == "FAILED" || "$STATE" == "ERROR" ]]; then
    echo "ERROR: 模型注册失败，任务详情："
    echo "$TASK_RESP"
    kill "$HTTP_SERVER_PID" || true
    exit 1
  else
    echo "当前任务状态: ${STATE:-unknown}，${POLL_INTERVAL}s 后重试..."
    sleep "$POLL_INTERVAL"
  fi
done

if [[ -z "$MODEL_ID" || "$MODEL_ID" == "null" ]]; then
  echo "ERROR: 任务完成但没有拿到 model_id，任务详情："
  echo "$TASK_RESP"
  kill "$HTTP_SERVER_PID" || true
  exit 1
fi

echo "[Info] 模型注册完成，model_id = ${MODEL_ID}"

########################################
# 7. 部署模型
########################################

echo_title "7. 部署模型 (model_id=${MODEL_ID})"

DEPLOY_RESP=$(curl_os -X POST "${OS_HOST}/_plugins/_ml/models/${MODEL_ID}/_deploy")
echo "$DEPLOY_RESP" | jq . || true

DEPLOY_TASK_ID=$(echo "$DEPLOY_RESP" | jq -r '.task_id // empty')
if [[ -z "$DEPLOY_TASK_ID" || "$DEPLOY_TASK_ID" == "null" ]]; then
  echo "ERROR: 部署模型时没有拿到 task_id，返回内容如下："
  echo "$DEPLOY_RESP"
  kill "$HTTP_SERVER_PID" || true
  exit 1
fi

echo "[Info] 部署任务 task_id = ${DEPLOY_TASK_ID}"

echo_title "8. 等待模型部署完成 (task_id=${DEPLOY_TASK_ID})"

while true; do
  DEPLOY_TASK_RESP=$(curl_os "${OS_HOST}/_plugins/_ml/tasks/${DEPLOY_TASK_ID}")
  echo "$DEPLOY_TASK_RESP" | jq . || true

  STATE=$(echo "$DEPLOY_TASK_RESP" | jq -r '.state // empty')
  if [[ "$STATE" == "COMPLETED" ]]; then
    echo "[Info] 模型部署完成。"
    break
  elif [[ "$STATE" == "FAILED" || "$STATE" == "ERROR" ]]; then
    echo "ERROR: 模型部署失败，任务详情："
    echo "$DEPLOY_TASK_RESP"
    kill "$HTTP_SERVER_PID" || true
    exit 1
  else
    echo "当前部署状态: ${STATE:-unknown}，${POLL_INTERVAL}s 后重试..."
    sleep "$POLL_INTERVAL"
  fi
done

kill "$HTTP_SERVER_PID" || true
echo_title "脚本执行完毕：模型已注册并部署，可通过 TEXT_EMBEDDING 使用。"
