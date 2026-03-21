#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/grouped_final_memory
RESULTS=results/raw/grouped_final_memory_results.tsv
LOG_DIR=runs/grouped_final_memory
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/grouped_final_memory_results.tsv').write_text('\t'.join([
  'config','seed','reference_lr','groups','value_scale','gate_cap','gate_bias_init',
  'val_bpb','num_steps','attnres_final_memory_gate','attnres_final_memory_gate_std','attnres_final_memory_delta_norm','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" lr="$3" groups="$4" value_scale="$5" gate_cap="$6" gate_bias_init="$7" log_path="$8"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${lr@Q}, ${groups@Q}, ${value_scale@Q}, ${gate_cap@Q}, ${gate_bias_init@Q},
  str(metrics.get('val_bpb','')), str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_memory_gate','')),
  str(metrics.get('attnres_final_memory_gate_std','')),
  str(metrics.get('attnres_final_memory_delta_norm','')),
  str(metrics.get('peak_vram_mb','')),
]))
PY
}

run_control() {
  local name="$1" seed="$2"
  local lr="0.0012"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  scripts/train_attnres.sh
  parse_log "$name" "$seed" "$lr" "0" "0.0" "0.0" "0.0" "$log_path"
}

run_static() {
  local name="$1" seed="$2" groups="$3" value_scale="$4" gate_cap="$5" gate_bias_init="$6"
  local lr="0.0012"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$value_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$gate_cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="$gate_bias_init" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$lr" "$groups" "$value_scale" "$gate_cap" "$gate_bias_init" "$log_path"
}

for seed in 45 46 47 48; do
  run_control strong_baseline "$seed"
  run_static static_g1_s025_c085 "$seed" 1 0.2500 0.85 4.0
  run_static static_g2_s025_c085 "$seed" 2 0.2500 0.85 4.0
  run_static static_g4_s025_c085 "$seed" 4 0.2500 0.85 4.0
  run_static static_g2_s03125_c085 "$seed" 2 0.3125 0.85 4.0
  run_static static_g4_s03125_c085 "$seed" 4 0.3125 0.85 4.0
done
