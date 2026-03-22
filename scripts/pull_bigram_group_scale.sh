#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs
REMOTE_ROOT=${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}
scp autodl:"$REMOTE_ROOT/results/raw/bigram_vram_probe_results.tsv" results/raw/bigram_vram_probe_results.tsv || true
scp autodl:"$REMOTE_ROOT/results/raw/bigram_group_scale_results.tsv" results/raw/bigram_group_scale_results.tsv || true
scp autodl:"$REMOTE_ROOT/runs/bigram_vram_probe.queue.log" runs/bigram_vram_probe.queue.log || true
scp autodl:"$REMOTE_ROOT/runs/bigram_group_scale.queue.log" runs/bigram_group_scale.queue.log || true
scp -r autodl:"$REMOTE_ROOT/runs/bigram_vram_probe" runs/ || true
scp -r autodl:"$REMOTE_ROOT/runs/bigram_group_scale" runs/ || true
