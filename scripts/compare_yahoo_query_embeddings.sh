#!/usr/bin/env bash
set -euo pipefail

# Wrapper for compare_query_embeddings.py that aligns vectors by query text.

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_FP16="$WORK_HOME/scripts/output/embeddings/yahoo_vecs_fp16.jsonl"
DEFAULT_FP32="$WORK_HOME/scripts/output/embeddings/yahoo_vecs_fp32.jsonl"

FILE_A="${FILE_A:-$DEFAULT_FP16}"
FILE_B="${FILE_B:-$DEFAULT_FP32}"
TOP_K="${TOP_K:-10}"
TARGET_DTYPE="${TARGET_DTYPE:-float32}"

if [[ $# -ge 1 ]]; then
  FILE_A="$1"
fi
if [[ $# -ge 2 ]]; then
  FILE_B="$2"
fi
if [[ $# -ge 3 ]]; then
  TOP_K="$3"
fi
if [[ $# -ge 4 ]]; then
  TARGET_DTYPE="$4"
fi

if [[ -z "$FILE_A" || -z "$FILE_B" ]]; then
  echo "Usage: $0 <file_a> <file_b> [top_k] [target_dtype]" >&2
  echo "  or configure FILE_A/FILE_B/TOP_K/TARGET_DTYPE env vars" >&2
  exit 1
fi

cat <<EOF
Comparing query-aligned embeddings:
  file_a       = $FILE_A
  file_b       = $FILE_B
  top_k        = $TOP_K
  target_dtype = $TARGET_DTYPE
EOF

echo
python "$WORK_HOME/src/embedding/compare_query_embeddings.py" \
  --file-a "$FILE_A" \
  --file-b "$FILE_B" \
  --top-k "$TOP_K" \
  --target-dtype "$TARGET_DTYPE"
