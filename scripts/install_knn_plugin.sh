#!/usr/bin/env bash
#
# install_knn.sh
#
# 一键编译并安装 OpenSearch k-NN 插件 + 编译 JNI so 的脚本。
# 使用前请根据实际环境修改下面的配置变量。
#

set -euo pipefail

########################################
#            配置区域（必改）           #
########################################

# k-NN 源码目录
KNN_SRC_DIR="/mnt/nvme2n1p1/xtang/opensearch-playground/k-NN"

# OpenSearch 解压后的根目录（里面应该有 bin/opensearch 和 bin/opensearch-plugin）
OPENSEARCH_DIR="/mnt/nvme2n1p1/xtang/opensearch-3.4.0-SNAPSHOT"

# OpenSearch 版本（需要和编译 OpenSearch 时一致）
OPENSEARCH_VERSION="3.4.0-SNAPSHOT"

# OpenSearch HTTP 地址（用于用 curl 做简单检查）
OPENSEARCH_URL="${OPENSEARCH_URL:-http://localhost:9200}"

# JNI so 部署目标目录：必须在 OpenSearch 的 java.library.path 中
# 你当前环境用的是 /home/opensearch/miniforge3/lib
JNI_TARGET_DIR="/home/opensearch/miniforge3/lib"

########################################
#         下面一般不需要修改            #
########################################

echo "========== [1/5] 环境检查 =========="

if [[ ! -d "${KNN_SRC_DIR}" ]]; then
  echo "ERROR: KNN_SRC_DIR 不存在: ${KNN_SRC_DIR}"
  exit 1
fi

if [[ ! -d "${OPENSEARCH_DIR}" ]]; then
  echo "ERROR: OPENSEARCH_DIR 不存在: ${OPENSEARCH_DIR}"
  exit 1
fi

if [[ ! -x "${KNN_SRC_DIR}/gradlew" ]]; then
  echo "ERROR: 在 ${KNN_SRC_DIR} 中找不到 gradlew 或不可执行"
  exit 1
fi

if [[ ! -x "${OPENSEARCH_DIR}/bin/opensearch-plugin" ]]; then
  echo "ERROR: 在 ${OPENSEARCH_DIR}/bin 中找不到 opensearch-plugin 工具"
  exit 1
fi

echo "KNN_SRC_DIR      = ${KNN_SRC_DIR}"
echo "OPENSEARCH_DIR   = ${OPENSEARCH_DIR}"
echo "OPENSEARCH_VER   = ${OPENSEARCH_VERSION}"
echo "OPENSEARCH_URL   = ${OPENSEARCH_URL}"
echo "JNI_TARGET_DIR   = ${JNI_TARGET_DIR}"
echo

# 提示一下 gfortran（Faiss CMake 会用到）
if ! command -v gfortran >/dev/null 2>&1; then
  echo "WARNING: 未找到 gfortran，JNI 编译可能失败。请先安装 gfortran（例如：sudo apt-get install -y gfortran）。"
fi

########################################
#   [2/5] 编译 JNI so 并拷贝到运行目录  #
########################################

echo "========== [2/5] 编译 JNI so (libopensearchknn_common.so) =========="

pushd "${KNN_SRC_DIR}/jni" >/dev/null

# 重新创建 build 目录（可按需保留旧构建，这里选择干净重建）
rm -rf build
mkdir -p build
cd build

echo "[JNI] 运行 CMake 配置..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKNN_PLUGIN_VERSION="${OPENSEARCH_VERSION}" \
  -DAPPLY_LIB_PATCHES=true \
  -DCOMMIT_LIB_PATCHES=true \
  -DBUILD_TESTING=OFF

echo "[JNI] 开始构建（不编译测试）..."
cmake --build . --config Release -j"$(nproc)"

echo "[JNI] 查找生成的 libopensearchknn_common.so ..."
JNI_SO_PATH=$(find "$(pwd)" -maxdepth 3 -type f -name "libopensearchknn_common*.so" | head -n 1 || true)

if [[ -z "${JNI_SO_PATH}" ]]; then
  echo "ERROR: 未在 JNI build 目录中找到 libopensearchknn_common*.so"
  popd >/dev/null
  exit 1
fi

echo "[JNI] 找到 JNI so: ${JNI_SO_PATH}"

# 确保目标目录存在
mkdir -p "${JNI_TARGET_DIR}"

echo "[JNI] 拷贝 JNI so 到 ${JNI_TARGET_DIR} ..."
cp "${JNI_SO_PATH}" "${JNI_TARGET_DIR}/"

echo "[JNI] JNI so 部署完成: ${JNI_TARGET_DIR}/$(basename "${JNI_SO_PATH}")"

popd >/dev/null
echo

########################################
#       [3/5] 编译 k-NN 插件           #
########################################

echo "========== [3/5] 编译 k-NN 插件 ZIP =========="

pushd "${KNN_SRC_DIR}" >/dev/null

# 使用本地 mavenLocal 中的 build-tools（之前由 OpenSearch 构建时发布）
./gradlew assemble \
  -Dopensearch.version="${OPENSEARCH_VERSION}" \
  -Dbuild.snapshot=true

echo "k-NN assemble 完成，查找构建好的插件 zip..."

KNN_ZIP=$(find build -maxdepth 3 -type f -name "opensearch-knn-*.zip" | head -n 1 || true)

if [[ -z "${KNN_ZIP}" ]]; then
  echo "ERROR: 未在 ${KNN_SRC_DIR}/build 下找到 opensearch-knn-*.zip"
  popd >/dev/null
  exit 1
fi

KNN_ZIP_ABS="$(cd "$(dirname "${KNN_ZIP}")" && pwd)/$(basename "${KNN_ZIP}")"

echo "找到 k-NN 插件包: ${KNN_ZIP_ABS}"
popd >/dev/null
echo

########################################
#       [4/5] 安装 k-NN 插件           #
########################################

echo "========== [4/5] 安装 k-NN 插件到 OpenSearch =========="

pushd "${OPENSEARCH_DIR}" >/dev/null

echo "当前已安装插件列表："
./bin/opensearch-plugin list || true
echo

echo "开始安装插件，可能会有交互提示（如是否继续安装）..."
./bin/opensearch-plugin install "file://${KNN_ZIP_ABS}"

echo
echo "安装完成后的插件列表："
./bin/opensearch-plugin list || true

popd >/dev/null
echo

########################################
#   [5/5] 通过 REST API 简单检查       #
########################################

echo "========== [5/5] 通过 REST API 检查插件状态 =========="

if command -v curl >/dev/null 2>&1; then
  echo "检查 OpenSearch 是否在 ${OPENSEARCH_URL} 运行..."

  if curl -s "${OPENSEARCH_URL}" >/dev/null 2>&1; then
    echo
    echo "1) _cat/plugins:"
    curl -s "${OPENSEARCH_URL}/_cat/plugins?v" || true

    echo
    echo "2) _nodes/plugins 中的 opensearch-knn（截断显示）："
    curl -s "${OPENSEARCH_URL}/_nodes/plugins?pretty" \
      | sed -n '1,200p' || true

    echo
    echo "3) k-NN stats（如果插件加载成功且支持该接口）："
    curl -s "${OPENSEARCH_URL}/_plugins/_knn/stats?pretty" \
      | sed -n '1,200p' || true

  else
    echo "WARNING: 无法访问 ${OPENSEARCH_URL}，可能是 OpenSearch 尚未启动，跳过 REST 检查。"
  fi
else
  echo "WARNING: 系统未找到 curl，跳过 REST 检查。"
fi

echo
echo "========== k-NN 插件安装脚本执行结束 =========="
echo "如果需要，请手动启动 OpenSearch："
echo "  cd \"${OPENSEARCH_DIR}\" && ./bin/opensearch"
