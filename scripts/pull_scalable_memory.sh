#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs

REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}
scp autodl:"$REMOTE_ROOT/results/raw/scalable_memory_results.tsv" results/raw/scalable_memory_results.tsv
scp autodl:"$REMOTE_ROOT/runs/scalable_memory.queue.log" runs/scalable_memory.queue.log || true
scp -r autodl:"$REMOTE_ROOT/runs/scalable_memory" runs/ || true
