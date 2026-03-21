#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

bash scripts/repro_main.sh
"$PYTHON_BIN" scripts/plot_main.py

bash scripts/repro_ablation.sh
"$PYTHON_BIN" scripts/plot_ablation.py

bash scripts/repro_followup.sh
"$PYTHON_BIN" scripts/plot_followup.py
