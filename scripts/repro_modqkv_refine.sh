#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/modqkv_refine
RESULTS=results/raw/modqkv_refine_results.tsv
LOG_DIR=runs/modqkv_refine
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/modqkv_refine_results.tsv').write_text('\t'.join([
  'config','seed','mode','source','banks','bigram_buckets','value_scale','q_mod_scale','k_mod_scale','v_mod_scale',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_mod_q_delta_norm','attnres_final_memory_mod_k_delta_norm','attnres_final_memory_mod_v_delta_norm',
  'attnres_final_memory_mod_k_scale_mean','attnres_final_memory_mod_v_scale_mean',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm',
  'attnres_final_memory_q_mod_proj_grad_norm','attnres_final_memory_k_mod_proj_grad_norm','attnres_final_memory_v_mod_proj_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" source="$4" banks="$5" buckets="$6" scale="$7" qscale="$8" kscale="$9" vscale="${10}" log_path="${11}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_mod_q_delta_norm','attnres_final_memory_mod_k_delta_norm','attnres_final_memory_mod_v_delta_norm',
  'attnres_final_memory_mod_k_scale_mean','attnres_final_memory_mod_v_scale_mean',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm',
  'attnres_final_memory_q_mod_proj_grad_norm','attnres_final_memory_k_mod_proj_grad_norm','attnres_final_memory_v_mod_proj_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${source@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${qscale@Q}, ${kscale@Q}, ${vscale@Q}]
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
  AUTORESEARCH_DEVICE_BATCH_SIZE="${AUTORESEARCH_DEVICE_BATCH_SIZE:-32}" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="${AUTORESEARCH_TOTAL_BATCH_SIZE:-16384}" \
  scripts/train_attnres.sh
  parse_log "strong_baseline" "$seed" "off" "off" "0" "0" "0.0" "0.0" "0.0" "0.0" "$log_path"
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
  AUTORESEARCH_DEVICE_BATCH_SIZE="${AUTORESEARCH_DEVICE_BATCH_SIZE:-32}" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="${AUTORESEARCH_TOTAL_BATCH_SIZE:-16384}" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="static" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="0.85" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="1048576" \
  scripts/train_final_memory_static.sh
  parse_log "blend_ngram_ref" "$seed" "static" "tied_bigram" "4" "1048576" "0.25" "0.0" "0.0" "0.0" "$log_path"
}

run_variant() {
  local name="$1" seed="$2" qscale="$3" kscale="$4" vscale="$5"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_DEVICE_BATCH_SIZE="${AUTORESEARCH_DEVICE_BATCH_SIZE:-32}" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="${AUTORESEARCH_TOTAL_BATCH_SIZE:-16384}" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="modqkv" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="1048576" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_Q_MOD_SCALE="$qscale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE="$kscale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE="$vscale" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "modqkv" "tied_bigram" "4" "1048576" "0.25" "$qscale" "$kscale" "$vscale" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_blend_ref "$seed"
  run_variant modqkv_q001_k002_v002 "$seed" 0.01 0.02 0.02
  run_variant modqkv_q002_k005_v005 "$seed" 0.02 0.05 0.05
  run_variant modqkv_q002_k005_v002 "$seed" 0.02 0.05 0.02
  run_variant modqkv_q002_k008_v002 "$seed" 0.02 0.08 0.02
  run_variant modqkv_q003_k008_v002 "$seed" 0.03 0.08 0.02
  run_variant modqkv_q003_k010_v002 "$seed" 0.03 0.10 0.02
  run_variant modqkv_q004_k010_v002 "$seed" 0.04 0.10 0.02
done
