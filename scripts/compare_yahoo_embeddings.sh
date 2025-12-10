#!/usr/bin/env bash
set -euo pipefail

# Wrapper script for comparing two yahoo_vecs.jsonl embedding files

WOKR_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FILE_A="${FILE_A:-}"
FILE_B="${FILE_B:-}"
TOP_K="${TOP_K:-10}"

FILE_A="$WOKR_HOME/scripts/output/embeddings/yahoo_vecs_fp16.jsonl"
FILE_B="$WOKR_HOME/scripts/output/embeddings/yahoo_vecs_fp32.jsonl"

if [[ $# -ge 2 ]]; then
  FILE_A="$1"
  FILE_B="$2"
fi

if [[ -z "$FILE_A" || -z "$FILE_B" ]]; then
  echo "Usage: $0 <file_a.jsonl> <file_b.jsonl> [top_k]" >&2
  echo "  or set FILE_A/FILE_B/TOP_K env vars before running" >&2
  exit 1
fi

if [[ $# -ge 3 ]]; then
  TOP_K="$3"
fi

echo "Comparing embeddings:"
echo "  file_a = $FILE_A"
echo "  file_b = $FILE_B"
echo "  top_k  = $TOP_K"

echo
python $WOKR_HOME/src/embedding/compare_yahoo_embeddings.py \
  --file-a "$FILE_A" \
  --file-b "$FILE_B" \
  --top-k "$TOP_K"