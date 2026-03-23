#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/memory_sources
RESULTS=results/raw/memory_sources_results.tsv
LOG_DIR=runs/memory_sources
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/memory_sources_results.tsv').write_text('\t'.join([
  'config','seed','reference_lr','source','rank','value_scale','gate_cap','gate_bias_init',
  'val_bpb','num_steps','attnres_final_memory_gate','attnres_final_memory_gate_std','attnres_final_memory_delta_norm','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" lr="$3" source="$4" rank="$5" value_scale="$6" gate_cap="$7" gate_bias_init="$8" log_path="$9"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${lr@Q}, ${source@Q}, ${rank@Q}, ${value_scale@Q}, ${gate_cap@Q}, ${gate_bias_init@Q},
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
  parse_log "$name" "$seed" "$lr" "off" "0" "0.0" "0.0" "0.0" "$log_path"
}

run_static_source() {
  local name="$1" seed="$2" source="$3" rank="$4" value_scale="$5" gate_cap="$6" gate_bias_init="$7"
  local lr="0.0012"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK="$rank" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$value_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$gate_cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="$gate_bias_init" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$lr" "$source" "$rank" "$value_scale" "$gate_cap" "$gate_bias_init" "$log_path"
}

for seed in 45 46 47 48; do
  run_control strong_baseline "$seed"
  run_static_source static_full_s025_c085 "$seed" full 0 0.2500 0.85 4.0
  run_static_source static_factorized_r32_s025_c085 "$seed" factorized 32 0.2500 0.85 4.0
  run_static_source static_factorized_r64_s025_c085 "$seed" factorized 64 0.2500 0.85 4.0
  run_static_source static_factorized_r128_s025_c085 "$seed" factorized 128 0.2500 0.85 4.0
  run_static_source static_tied_s025_c085 "$seed" tied 0 0.2500 0.85 4.0
done
