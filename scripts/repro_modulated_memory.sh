#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/modulated_memory
RESULTS=results/raw/modulated_memory_results.tsv
LOG_DIR=runs/modulated_memory
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/modulated_memory_results.tsv').write_text('\t'.join([
  'config','seed','mode','source','groups','banks','bigram_buckets','value_scale','q_mod_scale','k_mod_scale','v_mod_scale',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_mod_q_delta_norm','attnres_final_memory_mod_k_delta_norm','attnres_final_memory_mod_v_delta_norm',
  'attnres_final_memory_mod_k_scale_mean','attnres_final_memory_mod_v_scale_mean',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm',
  'attnres_final_memory_q_mod_proj_grad_norm','attnres_final_memory_k_mod_proj_grad_norm','attnres_final_memory_v_mod_proj_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" source="$4" groups="$5" banks="$6" buckets="$7" scale="$8" qscale="$9" kscale="${10}" vscale="${11}" log_path="${12}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_rms','attnres_final_memory_mod_q_delta_norm','attnres_final_memory_mod_k_delta_norm','attnres_final_memory_mod_v_delta_norm',
  'attnres_final_memory_mod_k_scale_mean','attnres_final_memory_mod_v_scale_mean',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm',
  'attnres_final_memory_q_mod_proj_grad_norm','attnres_final_memory_k_mod_proj_grad_norm','attnres_final_memory_v_mod_proj_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${source@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${qscale@Q}, ${kscale@Q}, ${vscale@Q}]
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
  parse_log "strong_baseline" "$seed" "off" "off" "0" "0" "0" "0.0" "0.0" "0.0" "0.0" "$log_path"
}

run_variant() {
  local name="$1" seed="$2" mode="$3" source="$4" groups="$5" banks="$6" buckets="$7" scale="$8" qscale="$9" kscale="${10}" vscale="${11}"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="$mode" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_Q_MOD_SCALE="$qscale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE="$kscale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE="$vscale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="0.85" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$mode" "$source" "$groups" "$banks" "$buckets" "$scale" "$qscale" "$kscale" "$vscale" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_variant blend_ngram_ref "$seed" static tied_bigram 4 4 1048576 0.25 0.0 0.0 0.0
  run_variant modkv_kv002 "$seed" modkv tied_bigram 1 4 1048576 0.25 0.0 0.02 0.02
  run_variant modkv_kv005 "$seed" modkv tied_bigram 1 4 1048576 0.25 0.0 0.05 0.05
  run_variant modkv_kv010 "$seed" modkv tied_bigram 1 4 1048576 0.25 0.0 0.10 0.10
  run_variant modqkv_q001_kv002 "$seed" modqkv tied_bigram 1 4 1048576 0.25 0.01 0.02 0.02
  run_variant modqkv_q002_kv005 "$seed" modqkv tied_bigram 1 4 1048576 0.25 0.02 0.05 0.05
done
