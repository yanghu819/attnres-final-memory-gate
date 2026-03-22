#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/vres_memory
RESULTS=results/raw/vres_memory_results.tsv
LOG_DIR=runs/vres_memory
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/vres_memory_results.tsv').write_text('\t'.join([
  'config','seed','mode','source','banks','bigram_buckets','value_mode','memory_scale','vres_scale',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_vres_delta_norm',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm','attnres_final_memory_vres_proj_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" source="$4" banks="$5" buckets="$6" value_mode="$7" memory_scale="$8" vres_scale="$9" log_path="${10}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_vres_delta_norm',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm','attnres_final_memory_vres_proj_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${source@Q}, ${banks@Q}, ${buckets@Q}, ${value_mode@Q}, ${memory_scale@Q}, ${vres_scale@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_control() {
  local seed="$1"
  local log_path="$LOG_DIR/strong_baseline_seed${seed}.log"
  echo "== strong_baseline_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  scripts/train_attnres.sh
  parse_log "strong_baseline" "$seed" "off" "off" "0" "0" "raw" "0.0" "0.0" "$log_path"
}

run_variant() {
  local name="$1" seed="$2" source="$3" banks="$4" buckets="$5" value_mode="$6" memory_scale="$7" vres_scale="$8"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="vres" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="$value_mode" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$memory_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VRES_SCALE="$vres_scale" \
  scripts/train_final_memory_vres.sh
  parse_log "$name" "$seed" "vres" "$source" "$banks" "$buckets" "$value_mode" "$memory_scale" "$vres_scale" "$log_path"
}

run_blend_ref() {
  local seed="$1"
  local log_path="$LOG_DIR/blend_ngram_ref_seed${seed}.log"
  echo "== blend_ngram_ref_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="static" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="0.85" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="1048576" \
  scripts/train_final_memory_static.sh
  parse_log "blend_ngram_ref" "$seed" "static" "tied_bigram" "4" "1048576" "rmsnorm" "0.25" "0.0" "$log_path"
}

run_modqkv_ref() {
  local seed="$1"
  local log_path="$LOG_DIR/modqkv_q002_kv005_seed${seed}.log"
  echo "== modqkv_q002_kv005_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="modqkv" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_Q_MOD_SCALE="0.02" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE="0.05" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE="0.05" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="1048576" \
  scripts/train_final_memory_static.sh
  parse_log "modqkv_q002_kv005" "$seed" "modqkv" "tied_bigram" "4" "1048576" "rmsnorm" "0.25" "0.0" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_blend_ref "$seed"
  run_modqkv_ref "$seed"
  run_variant vres_tied_a001 "$seed" tied 0 0 rmsnorm 1.0 0.01
  run_variant vres_tied_a002 "$seed" tied 0 0 rmsnorm 1.0 0.02
  run_variant vres_tied_a005 "$seed" tied 0 0 rmsnorm 1.0 0.05
  run_variant vres_tied_bigram_a002 "$seed" tied_bigram 4 1048576 rmsnorm 1.0 0.02
done
