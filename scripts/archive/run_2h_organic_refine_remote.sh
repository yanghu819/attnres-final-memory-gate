#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
RUN_ID="organic_refine_2h_$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_LOG="$ROOT/runs/${RUN_ID}.launcher.log"
REMOTE_QUEUE_LOG="$REMOTE_ROOT/runs/${RUN_ID}.queue.log"

mkdir -p "$ROOT/runs" "$ROOT/results/raw"

nohup bash -lc '
set -euo pipefail
while ssh '"$REMOTE_HOST"' "pgrep -f repro_modulated_memory.sh >/dev/null || pgrep -f repro_modulated_logit_memory.sh >/dev/null"; do
  sleep 30
done
cd '"$ROOT"'
rsync -az --delete-after \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude "results/*" \
  --exclude "runs/*" \
  ./ '"$REMOTE_HOST:$REMOTE_ROOT/"'
ssh '"$REMOTE_HOST"' "cd '"$REMOTE_ROOT"' && mkdir -p runs results/raw && PYTHONPATH=. /root/miniconda3/bin/python -m unittest tests.test_attnres_projected && timeout 7600 bash scripts/repro_organic_refine.sh > '"$REMOTE_QUEUE_LOG"' 2>&1"
scp '"$REMOTE_HOST:$REMOTE_ROOT/results/raw/organic_refine_results.tsv"' '"$ROOT/results/raw/organic_refine_results.tsv"' || true
scp '"$REMOTE_HOST:$REMOTE_QUEUE_LOG"' '"$ROOT/runs/${RUN_ID}.queue.log"' || true
scp -r '"$REMOTE_HOST:$REMOTE_ROOT/runs/organic_refine"' '"$ROOT/runs/"' || true
' > "$LOCAL_LOG" 2>&1 &
PID=$!
echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOCAL_LOG=$LOCAL_LOG"
echo "REMOTE_QUEUE_LOG=$REMOTE_QUEUE_LOG"
