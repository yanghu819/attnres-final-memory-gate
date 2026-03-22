#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
RUN_ID="ngram_module_ablation_2h_$(date -u +%Y%m%dT%H%M%SZ)"
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
  --exclude results/* \
  --exclude runs/* \
  ./ '$REMOTE_HOST:$REMOTE_ROOT/'
ssh '$REMOTE_HOST' \"cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && PYTHONPATH=. /root/miniconda3/bin/python tests/test_attnres_projected.py && timeout 7100 bash scripts/repro_ngram_module_ablation.sh > '$REMOTE_QUEUE_LOG' 2>&1 && PYTHONPATH=src /root/miniconda3/bin/python scripts/plot_ngram_module_ablation.py && /root/miniconda3/bin/python scripts/analyze_ngram_module_ablation.py\"
scp '$REMOTE_HOST:$REMOTE_ROOT/results/raw/ngram_module_ablation_results.tsv' '$ROOT/results/raw/ngram_module_ablation_results.tsv'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/figs/fig_ngram_module_ablation_loss_curves.png' '$ROOT/results/figs/fig_ngram_module_ablation_loss_curves.png'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/figs/fig_ngram_module_ablation_valbpb.png' '$ROOT/results/figs/fig_ngram_module_ablation_valbpb.png'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/ngram_module_ablation_results.tsv' '$ROOT/results/ngram_module_ablation_results.tsv'
scp '$REMOTE_HOST:$REMOTE_ROOT/results/ngram_module_ablation_analysis.txt' '$ROOT/results/ngram_module_ablation_analysis.txt'
scp '$REMOTE_HOST:$REMOTE_QUEUE_LOG' '$ROOT/runs/${RUN_ID}.queue.log' || true
scp -r '$REMOTE_HOST:$REMOTE_ROOT/runs/ngram_module_ablation' '$ROOT/runs/' || true
PYTHONPATH=src python3 scripts/plot_ngram_module_ablation.py
python3 scripts/analyze_ngram_module_ablation.py
" > "$LOCAL_LOG" 2>&1 &
PID=$!
echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOCAL_LOG=$LOCAL_LOG"
echo "REMOTE_QUEUE_LOG=$REMOTE_QUEUE_LOG"
