#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-/root/autodl-tmp/.cache/autoresearch}"
export AUTORESEARCH_BASE_URL="${AUTORESEARCH_BASE_URL:-https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main}"
export AUTORESEARCH_MAX_SEQ_LEN="${AUTORESEARCH_MAX_SEQ_LEN:-512}"
export AUTORESEARCH_EVAL_TOKENS="${AUTORESEARCH_EVAL_TOKENS:-1048576}"
export AUTORESEARCH_VOCAB_SIZE="${AUTORESEARCH_VOCAB_SIZE:-4096}"
NUM_SHARDS="${AUTORESEARCH_NUM_SHARDS:-2}"
DOWNLOAD_WORKERS="${AUTORESEARCH_DOWNLOAD_WORKERS:-4}"
exec /root/miniconda3/bin/python -m uv run prepare.py --num-shards "${NUM_SHARDS}" --download-workers "${DOWNLOAD_WORKERS}"
