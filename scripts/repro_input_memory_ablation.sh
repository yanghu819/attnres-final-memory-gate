#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/input_memory_ablation
RESULTS=results/raw/input_memory_ablation_results.tsv
LOG_DIR=runs/input_memory_ablation
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/input_memory_ablation_results.tsv').write_text('\t'.join([
  'config','seed','input_mode','smear','hash_dim','bigram_banks','bigram_buckets','trigram_banks','trigram_buckets',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_input_memory_bigram_rms','attnres_input_memory_trigram_rms','attnres_input_memory_smear_gate','attnres_input_memory_delta_norm',
  'attnres_input_memory_embed_grad_norm','attnres_input_memory_bigram_proj_grad_norm','attnres_input_memory_trigram_proj_grad_norm','attnres_input_memory_smear_gate_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" smear="$4" hdim="$5" bbanks="$6" bbuckets="$7" tbanks="$8" tbuckets="$9" log_path="${10}"
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
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${smear@Q}, ${hdim@Q}, ${bbanks@Q}, ${bbuckets@Q}, ${tbanks@Q}, ${tbuckets@Q}]
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
  parse_log "strong_baseline" "$seed" "off" "0" "0" "0" "0" "0" "0" "$log_path"
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
  parse_log "blend_ngram_ref" "$seed" "blend" "0" "0" "4" "1048576" "0" "0" "$log_path"
}

run_input() {
  local name="$1" seed="$2" mode="$3" smear="$4"
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
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE="1.0" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM="64" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS="16384" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE="1.0" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS="16384" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS="4" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE="1.0" \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT="-4.0" \
  scripts/train_attnres.sh
  if [[ "$mode" == "bigram_trigram" ]]; then
    parse_log "$name" "$seed" "$mode" "$smear" "64" "4" "16384" "4" "16384" "$log_path"
  else
    parse_log "$name" "$seed" "$mode" "$smear" "64" "4" "16384" "0" "0" "$log_path"
  fi
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_blend_ref "$seed"
  run_input input_bigram_hash "$seed" bigram 0
  run_input input_bigram_hash_smear "$seed" bigram 1
  run_input input_bigram_trigram_hash "$seed" bigram_trigram 0
done
