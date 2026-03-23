#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/work/autoresearch-attnres-project}"
REMOTE_HOST="${REMOTE_HOST:-autodl}"
RUN_ID="unified_memory_tokens_2h_$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_SCRIPT="$REMOTE_ROOT/runs/${RUN_ID}.sh"
REMOTE_QUEUE_LOG="$REMOTE_ROOT/runs/${RUN_ID}.queue.log"

mkdir -p "$ROOT/runs"

rsync -az --delete-after \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude "results/*" \
  --exclude "runs/*" \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"

ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && mkdir -p runs results/raw results/figs && cat > '$REMOTE_SCRIPT' <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
PYTHONPATH=. /root/miniconda3/bin/python tests/test_attnres_projected.py
bash scripts/repro_unified_memory_tokens.sh
PYTHONPATH=src /root/miniconda3/bin/python scripts/plot_unified_memory_tokens.py
PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_unified_memory_tokens.py
INNER
chmod +x '$REMOTE_SCRIPT'
nohup bash '$REMOTE_SCRIPT' > '$REMOTE_QUEUE_LOG' 2>&1 & echo RUN_ID='$RUN_ID' PID=\$! REMOTE_QUEUE_LOG='$REMOTE_QUEUE_LOG'"
