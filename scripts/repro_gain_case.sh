#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[gain-case] rerunning the historical winning case on fresh clean outputs"
bash scripts/repro_followup.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/plot_followup.py

echo "[gain-case] done"
