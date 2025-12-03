#!/usr/bin/env bash
set -euo pipefail

###############################################
# 配置：只需改这里
###############################################

# OpenSearch 安装目录（有 bin/opensearch 的目录）
OS_HOME="/mnt/nvme2n1p1/xtang/os-from-source/opensearch-epeshared-3.4.0-SNAPSHOT"

# plugin 源码所在目录（即 opensearch-playground）
PLAYGROUND="/mnt/nvme2n1p1/xtang/opensearch-playground"

###############################################
# 工具函数
###############################################

title() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

die() { echo "ERROR: $*" >&2; exit 1; }

###############################################
# 检查 OpenSearch 安装目录
###############################################

[[ -x "${OS_HOME}/bin/opensearch-plugin" ]] \
  || die "找不到 ${OS_HOME}/bin/opensearch-plugin，请检查 OS_HOME 是否正确。"

###############################################
# 查找插件 ZIP 文件
###############################################

title "1. 查找 opensearch-job-scheduler 插件 zip"

JOB_SCHEDULER_ZIP=$(find "${PLAYGROUND}/job-scheduler" -maxdepth 6 -type f -name "opensearch-job-scheduler-*.zip" | head -n1)
[[ -z "${JOB_SCHEDULER_ZIP}" ]] && die "未找到 job-scheduler ZIP，请确认 job-scheduler 已成功编译。"

echo "找到 job-scheduler 插件： ${JOB_SCHEDULER_ZIP}"


title "2. 查找 opensearch-ml 插件 zip"

ML_ZIP=$(find "${PLAYGROUND}/ml-commons" -maxdepth 8 -type f -name "opensearch-ml-*.zip" | head -n1)
[[ -z "${ML_ZIP}" ]] && die "未找到 opensearch-ml ZIP，请确认 ml-commons 已成功编译。"

echo "找到 ml-commons 插件： ${ML_ZIP}"

###############################################
# 提醒用户停止 OpenSearch
###############################################

title "请确保当前 OpenSearch 已被停止"
echo "例如：前台 Ctrl+C 或 systemd 停服务。"
echo "确认已停止后按回车继续…"
read -r _

###############################################
# 安装 job-scheduler
###############################################

title "3. 安装插件：opensearch-job-scheduler"

"${OS_HOME}/bin/opensearch-plugin" install "file://${JOB_SCHEDULER_ZIP}" || die "安装 opensearch-job-scheduler 失败"

echo "opensearch-job-scheduler 安装成功"

###############################################
# 安装 ml-commons (opensearch-ml)
###############################################

title "4. 安装插件：opensearch-ml"

"${OS_HOME}/bin/opensearch-plugin" install "file://${ML_ZIP}" || die "安装 opensearch-ml 失败"

echo "opensearch-ml 安装成功"

###############################################
# 启动提示
###############################################

title "插件安装完成！请重新启动 OpenSearch："
cat <<EOF
cd ${OS_HOME}
bin/opensearch

启动后验证插件是否加载：

  curl -sS http://localhost:9200/_cat/plugins?v

应看到：
  opensearch-job-scheduler
  opensearch-ml
EOF
