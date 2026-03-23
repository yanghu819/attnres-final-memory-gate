#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs

REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}
scp autodl:"$REMOTE_ROOT/results/raw/bigram_residual_refine_results.tsv" results/raw/bigram_residual_refine_results.tsv
scp autodl:"$REMOTE_ROOT/runs/bigram_residual_refine.queue.log" runs/bigram_residual_refine.queue.log || true
scp -r autodl:"$REMOTE_ROOT/runs/bigram_residual_refine" runs/ || true
