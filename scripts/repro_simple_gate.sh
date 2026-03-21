#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs/simple_gate
RESULTS=results/raw/simple_gate_results.tsv
LOG_DIR=runs/simple_gate
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/simple_gate_results.tsv').write_text('\t'.join([
  'config','seed','reference_lr','value_scale','gate_cap','gate_bias_init','delta_scale','val_bpb','num_steps',
  'attnres_final_register','attnres_final_memory_gate','attnres_final_memory_delta_norm','peak_vram_mb'
]) + '\n')
PY
parse_log() {
  local name="$1" seed="$2" lr="$3" value_scale="$4" gate_cap="$5" gate_bias_init="$6" delta_scale="$7" log_path="$8"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${lr@Q}, ${value_scale@Q}, ${gate_cap@Q}, ${gate_bias_init@Q}, ${delta_scale@Q},
  str(metrics.get('val_bpb','')), str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_register','')),
  str(metrics.get('attnres_final_memory_gate','')),
  str(metrics.get('attnres_final_memory_delta_norm','')),
  str(metrics.get('peak_vram_mb','')),
]))
PY
}
run_control() {
  local name="$1" seed="$2"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  local lr="0.0012"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  scripts/train_attnres.sh
  parse_log "$name" "$seed" "$lr" "0.0" "0.0" "0.0" "0.0" "$log_path"
}
run_projected_cap() {
  local name="$1" seed="$2"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  local lr="0.0012"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_REGISTER_SCOPE="final_only" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE="0.3125" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_START="0.3125" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS=0 \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_DELTA_SCALE="1.0" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_WEIGHT_CAP="0.85" \
  scripts/train_projected.sh
  parse_log "$name" "$seed" "$lr" "0.3125" "0.85" "0.0" "1.0" "$log_path"
}
run_final_memory() {
  local name="$1" preset="$2" seed="$3" value_scale="$4" gate_cap="$5" gate_bias_init="$6" delta_scale="$7"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  local lr="0.0012"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$value_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$gate_cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="$gate_bias_init" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_DELTA_SCALE="$delta_scale" \
  scripts/_autodl_env.sh train --preset "$preset"
  parse_log "$name" "$seed" "$lr" "$value_scale" "$gate_cap" "$gate_bias_init" "$delta_scale" "$log_path"
}
for seed in 45 46 47 48; do
  run_control strong_baseline "$seed"
  run_projected_cap projected_cap085 "$seed"
  run_final_memory final_memory_static final_memory_static "$seed" 0.3125 0.85 4.0 1.0
  run_final_memory final_memory_dynamic final_memory_dynamic "$seed" 0.3125 0.85 4.0 1.0
  run_final_memory final_memory_dynamic_cap090 final_memory_dynamic "$seed" 0.3125 0.90 4.0 1.0
done
