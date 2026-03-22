#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
RUN_ID="model_size_holdout_$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_LOG="$ROOT/runs/${RUN_ID}.launcher.log"
REMOTE_QUEUE_LOG="$REMOTE_ROOT/runs/${RUN_ID}.queue.log"

mkdir -p "$ROOT/runs"

{
  echo "RUN_ID=$RUN_ID"
  echo "REMOTE_HOST=$REMOTE_HOST"
  echo "REMOTE_ROOT=$REMOTE_ROOT"
  echo "REMOTE_QUEUE_LOG=$REMOTE_QUEUE_LOG"
} | tee "$LOCAL_LOG"

rsync -az --delete-after \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  "$ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/" | tee -a "$LOCAL_LOG"

ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && PYTHONPATH=. /root/miniconda3/bin/python tests/test_attnres_projected.py && timeout 24000 bash -lc 'bash scripts/repro_model_size_probe.sh && PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_model_size_probe.py && bash scripts/repro_model_size_holdout.sh && PYTHONPATH=src /root/miniconda3/bin/python scripts/plot_model_size_holdout.py && PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_model_size_holdout.py' > '$REMOTE_QUEUE_LOG' 2>&1" | tee -a "$LOCAL_LOG"

scp "$REMOTE_HOST:$REMOTE_QUEUE_LOG" "$ROOT/runs/${RUN_ID}.queue.log" || true

echo "RUN_ID=$RUN_ID"
echo "REMOTE_QUEUE_LOG=$REMOTE_QUEUE_LOG"
