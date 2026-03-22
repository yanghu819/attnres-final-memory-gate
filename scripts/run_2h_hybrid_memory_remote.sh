#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
RUN_ID="hybrid_memory_2h_$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_LOG="$ROOT/runs/${RUN_ID}.launcher.log"
REMOTE_QUEUE_LOG="$REMOTE_ROOT/runs/${RUN_ID}.queue.log"

mkdir -p "$ROOT/runs" "$ROOT/results/raw" "$ROOT/results/figs"

nohup bash -lc "
set -euo pipefail
cd '$ROOT'
rsync -az --delete-after \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude 'results/*' \
  --exclude 'runs/*' \
  ./ '$REMOTE_HOST:$REMOTE_ROOT/'
ssh '$REMOTE_HOST' \"cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && PYTHONPATH=. /root/miniconda3/bin/python tests/test_attnres_projected.py && timeout 10800 bash scripts/repro_hybrid_memory.sh > '$REMOTE_QUEUE_LOG' 2>&1 && PYTHONPATH=src /root/miniconda3/bin/python scripts/plot_hybrid_memory.py && PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_hybrid_memory.py\"
scp '$REMOTE_HOST:$REMOTE_ROOT/results/raw/hybrid_memory_results.tsv' '$ROOT/results/raw/hybrid_memory_results.tsv'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/figs/fig_hybrid_memory_loss_curves.png' '$ROOT/results/figs/fig_hybrid_memory_loss_curves.png'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/figs/fig_hybrid_memory_summary.png' '$ROOT/results/figs/fig_hybrid_memory_summary.png'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/hybrid_memory_results.tsv' '$ROOT/results/hybrid_memory_results.tsv'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/hybrid_memory_analysis.txt' '$ROOT/results/hybrid_memory_analysis.txt'
scp '$REMOTE_HOST:$REMOTE_QUEUE_LOG' '$ROOT/runs/${RUN_ID}.queue.log' || true
scp -r '$REMOTE_HOST:$REMOTE_ROOT/runs/hybrid_memory' '$ROOT/runs/' || true
" > "$LOCAL_LOG" 2>&1 &
PID=$!
echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOCAL_LOG=$LOCAL_LOG"
echo "REMOTE_QUEUE_LOG=$REMOTE_QUEUE_LOG"
