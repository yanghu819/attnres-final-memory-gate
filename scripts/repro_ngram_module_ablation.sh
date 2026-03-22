#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/ngram_module_ablation
RESULTS=results/raw/ngram_module_ablation_results.tsv
LOG_DIR=runs/ngram_module_ablation
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/ngram_module_ablation_results.tsv').write_text('\t'.join([
  'config','seed','source','groups','banks','bigram_buckets','value_scale','gate_cap','peak_vram_mb','val_bpb','num_steps',
  'attnres_final_memory_gate','attnres_final_memory_gate_std','attnres_final_memory_delta_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" source="$3" groups="$4" banks="$5" buckets="$6" scale="$7" cap="$8" log_path="$9"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${source@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${cap@Q},
  str(metrics.get('peak_vram_mb','')),
  str(metrics.get('val_bpb','')),
  str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_memory_gate','')),
  str(metrics.get('attnres_final_memory_gate_std','')),
  str(metrics.get('attnres_final_memory_delta_norm','')),
]))
PY
}

run_cfg() {
  local name="$1" seed="$2" source="$3" groups="$4" banks="$5" buckets="$6" scale="$7" cap="$8"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="1.0" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$seed" "$source" "$groups" "$banks" "$buckets" "$scale" "$cap" "$log_path"
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
  parse_log "strong_baseline" "$seed" "off" "0" "0" "0" "0.0" "0.0" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_cfg unigram_only_matched "$seed" tied 4 0 0 0.25 0.85
  run_cfg unigram_only_best "$seed" tied 4 0 0 0.25 0.80
  run_cfg bigram_only_4x1m "$seed" tied_bigram_only 4 4 1048576 0.25 0.85
  run_cfg unigram_plus_bigram_4x1m "$seed" tied_bigram 4 4 1048576 0.25 0.85

done
