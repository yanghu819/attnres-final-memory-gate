#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs
REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}
scp autodl:"$REMOTE_ROOT/results/raw/ngram_module_ablation_results.tsv" results/raw/ngram_module_ablation_results.tsv || true
scp autodl:"$REMOTE_ROOT/runs/ngram_module_ablation.queue.log" runs/ngram_module_ablation.queue.log || true
scp -r autodl:"$REMOTE_ROOT/runs/ngram_module_ablation" runs/ || true
