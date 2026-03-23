#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
rsync -az --delete-after \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude 'results/*' \
  --exclude 'runs/*' \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && timeout 7600 bash scripts/repro_input_memory_mainline.sh > runs/input_memory_mainline.queue.log 2>&1 && PYTHONPATH=src /root/miniconda3/bin/python scripts/plot_input_memory_mainline.py && PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_input_memory_mainline.py"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/raw/input_memory_mainline_results.tsv" "$ROOT/results/raw/input_memory_mainline_results.tsv"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/input_memory_mainline_results.tsv" "$ROOT/results/input_memory_mainline_results.tsv"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/input_memory_mainline_analysis.txt" "$ROOT/results/input_memory_mainline_analysis.txt"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/figs/fig_input_memory_mainline_valbpb.png" "$ROOT/results/figs/fig_input_memory_mainline_valbpb.png"
