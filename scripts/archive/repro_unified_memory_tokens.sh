#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/unified_memory_tokens
RESULTS=results/raw/unified_memory_tokens_results.tsv
LOG_DIR=runs/unified_memory_tokens
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/unified_memory_tokens_results.tsv').write_text('\t'.join([
  'config','seed','mode','source','groups','banks','bigram_buckets','value_scale','gate_cap','delta_scale',
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_memory_total','attnres_final_memory_unigram',
  'attnres_final_memory_bigram_bank0','attnres_final_memory_bigram_bank1','attnres_final_memory_bigram_bank2','attnres_final_memory_bigram_bank3',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm',
  'attnres_final_memory_gate_bias_grad_norm','attnres_final_memory_gate_proj_grad_norm',
  'attnres_final_memory_unified_bias_grad_norm','attnres_final_memory_unified_proj_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" mode="$3" source="$4" groups="$5" banks="$6" buckets="$7" scale="$8" cap="$9" delta_scale="${10}" log_path="${11}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_final_memory_total','attnres_final_memory_unigram',
  'attnres_final_memory_bigram_bank0','attnres_final_memory_bigram_bank1','attnres_final_memory_bigram_bank2','attnres_final_memory_bigram_bank3',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_wte_grad_norm','attnres_final_memory_embed_grad_norm','attnres_final_memory_proj_grad_norm',
  'attnres_final_memory_gate_bias_grad_norm','attnres_final_memory_gate_proj_grad_norm',
  'attnres_final_memory_unified_bias_grad_norm','attnres_final_memory_unified_proj_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${mode@Q}, ${source@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${cap@Q}, ${delta_scale@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_unified() {
  local name="$1" seed="$2" mode="$3" source="$4" groups="$5" banks="$6" buckets="$7" scale="$8" cap="$9" delta_scale="${10}"
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
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_DELTA_SCALE="$delta_scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="1.0" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$mode" "$source" "$groups" "$banks" "$buckets" "$scale" "$cap" "$delta_scale" "$log_path"
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
  parse_log "strong_baseline" "$seed" "off" "off" "0" "0" "0" "0.0" "0.0" "0.0" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_unified blend_ngram_ref "$seed" static tied_bigram 4 4 1048576 0.25 0.85 1.0
  run_unified unified_bigram_only_static "$seed" unified_static tied_bigram_only 1 4 1048576 0.25 0.85 1.0
  run_unified unified_bigram_only_projected "$seed" unified_projected tied_bigram_only 1 4 1048576 0.25 0.85 1.0
  run_unified unified_ngram_static "$seed" unified_static tied_bigram 1 4 1048576 0.25 0.85 1.0
  run_unified unified_ngram_projected "$seed" unified_projected tied_bigram 1 4 1048576 0.25 0.85 1.0

done
