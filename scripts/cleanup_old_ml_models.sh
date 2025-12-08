#!/usr/bin/env bash
#
# cleanup_old_ml_models_simple.sh
#
# 思路：
#   1. 用 TEXT_EMBEDDING 查询全部相关模型
#   2. 过滤掉 chunk 文档（_id 结尾是 "_数字" 的）
#   3. 按 model_group_id 分组，只保留最大 model_version
#   4. 旧版本：先 UNDEPLOY，再 DELETE
#
# 环境变量：
#   OS_URL      - OpenSearch 地址，默认 http://localhost:9200
#   DRY_RUN     - true/false，默认 true
#   NAME_FILTER - 可选：只处理 name 中包含该字符串的模型
#

set -euo pipefail

OS_URL="${OS_URL:-http://localhost:9200}"
DRY_RUN="${DRY_RUN:-true}"
NAME_FILTER="${NAME_FILTER:-}"

echo "== OpenSearch ML 模型清理脚本 (simple+undeploy) =="
echo "OS_URL      = ${OS_URL}"
echo "DRY_RUN     = ${DRY_RUN}"
echo "NAME_FILTER = ${NAME_FILTER}"
echo

########################################
# 1. 查询 TEXT_EMBEDDING 模型
########################################
echo "[1/3] 从 OpenSearch 拉取 TEXT_EMBEDDING 模型列表..."

RESP_JSON=$(curl -s -X POST "${OS_URL}/_plugins/_ml/models/_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "term": {
        "algorithm": {
          "value": "TEXT_EMBEDDING"
        }
      }
    },
    "size": 1000
  }')

# 兼容 "total": 27 和 "total": { "value": 27, ... }
TOTAL=$(echo "${RESP_JSON}" | jq '.hits.total.value? // .hits.total // 0')
echo "  -> 共匹配到 ${TOTAL} 条 TEXT_EMBEDDING 模型文档（包含各版本及 chunk 文档）"
echo

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "没有找到任何 TEXT_EMBEDDING 模型文档，直接退出。"
  exit 0
fi

########################################
# 2. 解析：去掉 chunk，按 group 选最新版本
########################################
echo "[2/3] 解析模型版本，计算需要保留/删除的模型版本..."

SUMMARY_JSON=$(echo "${RESP_JSON}" | jq \
  --arg name_filter "${NAME_FILTER}" '
  .hits.hits
  | map({
      id: ._id,
      name: ._source.name,
      group: (._source.model_group_id // ._source.name // "NO_GROUP"),
      version: (._source.model_version // ._source.version // "0"),
      state: ._source.model_state,
      algorithm: (._source.algorithm // "UNKNOWN")
    })
  # 排除 chunk 文档：_id 以 "_数字" 结尾
  | map(select(.id | test("_[0-9]+$") | not))
  # 按 name 过滤（可选）
  | (if $name_filter != "" then
       map(select(.name | tostring | contains($name_filter)))
     else . end)
  # 过滤掉没有 version 的
  | map(select(.version != null and .version != ""))
  # version 转成数字
  | map(.version_num = (.version | tonumber? // 0))
  # 按 group 分组
  | group_by(.group)
  # 每组保留最大 version，其余标为 delete
  | map(
      sort_by(.version_num)
      | {
          group: (.[-1].group),
          name: (.[-1].name),
          keep: .[-1],
          delete: (if length > 1 then .[0:-1] else [] end)
        }
    )
')

GROUP_COUNT=$(echo "${SUMMARY_JSON}" | jq 'length')
if [[ "${GROUP_COUNT}" -eq 0 ]]; then
  echo "过滤后没有任何需要处理的模型（可能是 NAME_FILTER 太严格），退出。"
  exit 0
fi

echo "------ 模型分组版本概要（每个 group 一行） ------"
echo "${SUMMARY_JSON}" | jq -r '
  .[]
  | "Group: \(.group)\n  Name: \(.name)\n  Keep version: \(.keep.version) (id=\(.keep.id), state=\(.keep.state), algo=\(.keep.algorithm))\n  Delete versions: \(
      if (.delete | length) == 0
      then "none"
      else (.delete | map("\(.version) (id=\(.id), state=\(.state), algo=\(.algorithm))") | join(", "))
      end
    )\n"
'
echo "------------------------------------------------"
echo

# 要删除的 model_id 列表
DELETE_IDS=$(echo "${SUMMARY_JSON}" | jq -r '
  .[]
  | .delete[]
  | .id
')

if [[ -z "${DELETE_IDS}" ]]; then
  echo "[3/3] 没有发现需要删除的旧版本（每个 group 只有一个版本），退出。"
  exit 0
fi

echo "[3/3] 以下 model_id 将被视为“旧版本”（会先 UNDEPLOY 再 DELETE）："
echo "${DELETE_IDS}" | sed 's/^/  - /'
echo

if [[ "${DRY_RUN}" != "false" ]]; then
  echo "当前为 DRY_RUN 模式，不会真正调用 undeploy/delete。"
  echo "如需执行删除，请设置环境变量：DRY_RUN=false 再运行本脚本，例如："
  echo "  DRY_RUN=false NAME_FILTER=\"msmarco-distilbert-base-tas-b\" ./cleanup_old_ml_models_simple.sh"
  exit 0
fi

########################################
# 3. 对每个旧版本：先 UNDEPLOY，再 DELETE
########################################
echo "开始对旧版本模型执行 UNDEPLOY + DELETE ..."

while read -r MID; do
  [[ -z "${MID}" ]] && continue

  echo "  -> UNDEPLOY model_id=${MID}"
  curl -s -X POST "${OS_URL}/_plugins/_ml/models/${MID}/_undeploy" \
       -H "Content-Type: application/json" | jq .

  echo "  -> DELETE model_id=${MID}"
  curl -s -X DELETE "${OS_URL}/_plugins/_ml/models/${MID}" | jq .

  echo
done <<< "${DELETE_IDS}"

echo "完成旧版本清理。"
