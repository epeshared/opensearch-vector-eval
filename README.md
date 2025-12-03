# opensearch-vector-eval

This repository contains small utilities to test and evaluate OpenSearch text embedding models on Yahoo Answers data.

## Prerequisites

- A running OpenSearch cluster with the ML Commons / OpenSearch-ML plugin installed
- `curl`, `jq`, and Python 3 available on your machine

## Scripts Overview

- `scripts/setup_embedding.sh` – Configure ML cluster settings, register a pretrained TEXT_EMBEDDING model, deploy it, and run a small sanity-check prediction.
- `scripts/test_text_embedding.sh` – Call the TEXT_EMBEDDING `_predict` API once with two sample sentences, pretty-print the JSON response, and save it to `output/embeddings/test_prediction.json`.
- `scripts/run_yahoo_embedding.sh` – Batch-embed Yahoo Answers questions and write results to `output/embeddings/yahoo_vecs.jsonl`.

## Installing and Deploying the Model

From the `scripts` directory:

```bash
cd scripts
./setup_embedding.sh
```

This script will:

1. Check cluster health and that the ML plugin is installed.
2. Configure basic ML-related cluster settings.
3. Register a pretrained TEXT_EMBEDDING model.
4. Wait for registration to finish and retrieve the `model_id`.
5. Deploy the model and wait until it is `COMPLETED`.
6. Run a single embedding request to verify the model works.

Environment variables you can override:

- `OS_HOST` – OpenSearch URL (default: `http://localhost:9200`)
- `OS_USER`, `OS_PASS` – Optional basic auth credentials

## Quick Embedding Test

After a model is deployed, you can run a quick manual prediction:

```bash
cd scripts
./test_text_embedding.sh
```

The script will automatically search OpenSearch for a deployed `TEXT_EMBEDDING` model and use its `model_id`. The raw JSON response is stored in:

- `output/embeddings/test_prediction.json`

You can change the output path by setting the `OUTPUT` environment variable before running the script.

## Embedding Yahoo Answers Data

To generate embeddings for the Yahoo Answers dataset:

```bash
cd scripts
./run_yahoo_embedding.sh
```

This script will:

1. Look up a deployed `TEXT_EMBEDDING` model in OpenSearch and obtain its `model_id`.
2. Call the Python script `src/yahoo_qeury_embedding.py` to batch-embed queries.
3. Write one JSON object per line to `output/embeddings/yahoo_vecs.jsonl` with fields `id`, `query`, and `embedding`.

Configurable parameters (via environment variables):

- `OS_URL` – OpenSearch URL (default: `http://localhost:9200`)
- `INPUT` – Path to the Yahoo Answers JSONL input file
- `OUTPUT` – Output file path for embeddings (default: `output/embeddings/yahoo_vecs.jsonl`)
- `BATCH_SIZE` – Batch size for embedding requests (default: `100`)
- `MAX_QUERIES` – If set, only the first N queries are embedded (useful for quick tests)

The underlying Python script prints total embedding time and QPS, as well as a simple progress bar while processing.

## Python Embedding Script

The main embedding logic lives in `src/yahoo_qeury_embedding.py`:

- Reads the Yahoo JSONL file and extracts question text.
- Batches queries and sends them to OpenSearch via `OpenSearchTextEmbedder` in `src/opensearch/os_embeeding_client.py`.
- Writes out per-query JSON lines with `id`, `query`, and `embedding`.

You can also invoke it directly, for example:

```bash
python src/yahoo_qeury_embedding.py \
  --input /path/to/yahoo_answers_title_answer.jsonl \
  --output output/embeddings/yahoo_vecs.jsonl \
  --os-url http://localhost:9200 \
  --model-id <your_model_id> \
  --batch-size 100 \
  --max-queries 1000
```

Optional arguments:

- `--username`, `--password` – Basic auth for OpenSearch
- `--insecure` – Disable TLS verification if needed

This README only covers the basic workflow; you can adjust scripts and parameters to fit your own datasets or models.
