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
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && timeout 7600 bash scripts/repro_lightweight_final_blend_verify.sh > runs/lightweight_final_blend_verify.queue.log 2>&1 && PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_lightweight_final_blend_verify.py"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/raw/lightweight_final_blend_verify_results.tsv" "$ROOT/results/raw/lightweight_final_blend_verify_results.tsv"
scp "$REMOTE_HOST:$REMOTE_ROOT/results/lightweight_final_blend_verify_analysis.txt" "$ROOT/results/lightweight_final_blend_verify_analysis.txt"
