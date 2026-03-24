#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 8*3600 ))}"
QUEUE_LOG="runs/golf_8h_final_lightblend.queue.log"
FILL_LOG="runs/golf_occupancy_fill.queue.log"
MAIN_PATTERN='repro_golf_8h_final_lightblend.sh'
FILL_PATTERN='repro_golf_occupancy_fill.sh'
while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do
  if ! ps -eo cmd | grep -F "$MAIN_PATTERN" | grep -v grep >/dev/null 2>&1 && \
     ! ps -eo cmd | grep -F "$FILL_PATTERN" | grep -v grep >/dev/null 2>&1; then
    echo "[$(date -Is)] launching occupancy fill until $DEADLINE_EPOCH" >> "$FILL_LOG"
    nohup env DEADLINE_EPOCH="$DEADLINE_EPOCH" bash scripts/repro_golf_occupancy_fill.sh >> "$FILL_LOG" 2>&1 &
  fi
  sleep 60
done
