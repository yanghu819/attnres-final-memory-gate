#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/input_memory_refine
RESULTS=results/raw/input_memory_refine_results.tsv
LOG_DIR=runs/input_memory_refine
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/input_memory_refine_results.tsv').write_text('\t'.join([
  'config','seed','input_mode','smear','input_scale','bigram_scale','trigram_scale','smear_bias_init',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_input_memory_bigram_rms','attnres_input_memory_trigram_rms','attnres_input_memory_smear_gate','attnres_input_memory_delta_norm',
  'attnres_input_memory_embed_grad_norm','attnres_input_memory_bigram_proj_grad_norm','attnres_input_memory_trigram_proj_grad_norm','attnres_input_memory_smear_gate_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" smear="$4" iscale="$5" bscale="$6" tscale="$7" sbias="$8" log_path="$9"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_input_memory_bigram_rms','attnres_input_memory_trigram_rms','attnres_input_memory_smear_gate','attnres_input_memory_delta_norm',
  'attnres_input_memory_embed_grad_norm','attnres_input_memory_bigram_proj_grad_norm','attnres_input_memory_trigram_proj_grad_norm','attnres_input_memory_smear_gate_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${smear@Q}, ${iscale@Q}, ${bscale@Q}, ${tscale@Q}, ${sbias@Q}]
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
  parse_log "strong_baseline" "$seed" "off" "0" "0" "0" "0" "0" "$log_path"
}

run_input() {
  local name="$1" seed="$2" mode="$3" smear="$4" iscale="$5" bscale="$6" tscale="$7" sbias="$8"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE="$mode" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR="$smear" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE="$iscale" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM="64" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS="16384" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE="$bscale" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS="16384" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE="$tscale" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT="$sbias" \
  scripts/train_attnres.sh
  parse_log "$name" "$seed" "$mode" "$smear" "$iscale" "$bscale" "$tscale" "$sbias" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_input input_bigram_smear_ref "$seed" bigram 1 1.0 1.0 1.0 -4.0
  run_input input_bigram_smear_s050 "$seed" bigram 1 0.5 1.0 1.0 -4.0
  run_input input_bigram_smear_s025 "$seed" bigram 1 0.25 1.0 1.0 -4.0
  run_input input_bigram_smear_s050_b2 "$seed" bigram 1 0.5 1.0 1.0 -2.0
  run_input input_bigram_trigram_t025_s050 "$seed" bigram_trigram 0 0.5 1.0 0.25 -4.0
  run_input input_bigram_trigram_t010_s050 "$seed" bigram_trigram 0 0.5 1.0 0.10 -4.0
done
