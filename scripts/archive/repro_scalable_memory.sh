#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/scalable_memory
RESULTS=results/raw/scalable_memory_results.tsv
LOG_DIR=runs/scalable_memory
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/scalable_memory_results.tsv').write_text('\t'.join([
  'config','seed','reference_lr','source','residual_rank','bigram_buckets',
  'value_scale','gate_cap','gate_bias_init','residual_scale','bigram_scale',
  'val_bpb','num_steps','attnres_final_memory_gate','attnres_final_memory_gate_std',
  'attnres_final_memory_delta_norm','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" lr="$3" source="$4" residual_rank="$5" bigram_buckets="$6" value_scale="$7" gate_cap="$8" gate_bias_init="$9" residual_scale="${10}" bigram_scale="${11}" log_path="${12}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${lr@Q}, ${source@Q}, ${residual_rank@Q}, ${bigram_buckets@Q},
  ${value_scale@Q}, ${gate_cap@Q}, ${gate_bias_init@Q}, ${residual_scale@Q}, ${bigram_scale@Q},
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
  parse_log "$name" "$seed" "$lr" "off" "0" "0" "0.0" "0.0" "0.0" "0.0" "0.0" "$log_path"
}

run_static_source() {
  local name="$1" seed="$2" source="$3" residual_rank="$4" bigram_buckets="$5" value_scale="$6" gate_cap="$7" gate_bias_init="$8" residual_scale="$9" bigram_scale="${10}"
  local lr="0.0012"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_RESIDUAL_RANK="$residual_rank" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$bigram_buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$value_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$gate_cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="$gate_bias_init" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_RESIDUAL_SCALE="$residual_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="$bigram_scale" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$lr" "$source" "$residual_rank" "$bigram_buckets" "$value_scale" "$gate_cap" "$gate_bias_init" "$residual_scale" "$bigram_scale" "$log_path"
}

for seed in 45 46 47 48; do
  run_control strong_baseline "$seed"
  run_static_source static_full_s025_c085 "$seed" full 0 0 0.2500 0.85 4.0 0.0 0.0
  run_static_source tied_s025_c080 "$seed" tied 0 0 0.2500 0.80 4.0 0.0 0.0
  run_static_source tied_residual_r64_s025_c080 "$seed" tied_residual 64 0 0.2500 0.80 4.0 1.0 0.0
  run_static_source tied_residual_r128_s025_c080 "$seed" tied_residual 128 0 0.2500 0.80 4.0 1.0 0.0
  run_static_source tied_bigram_b8192_s025_c080 "$seed" tied_bigram 0 8192 0.2500 0.80 4.0 0.0 1.0
  run_static_source tied_bigram_b16384_s025_c080 "$seed" tied_bigram 0 16384 0.2500 0.80 4.0 0.0 1.0
done
