#!/usr/bin/env bash
set -euo pipefail

WOKR_HOME="$PWD/../"
echo "WOKR_HOME: $WOKR_HOME"

python $WOKR_HOME/src/embedding/export_text_embedding_ts.py \
  --model-name sentence-transformers/msmarco-distilbert-base-tas-b \
  --output $WOKR_HOME/src/embedding/ts_models/sentence-transformers/msmarco_distilbert_base_ts.pt \
  --pooling mean \
  --max-seq-len 128 \
  --dtype float16 \