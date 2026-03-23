#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/model_size_holdout
RESULTS=results/raw/model_size_holdout_results.tsv
LOG_DIR=runs/model_size_holdout
SELECTED=results/model_size_selected_sizes.tsv
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

cat > "$RESULTS" <<'EOF'
size_tag	depth	head_dim	device_batch_size	total_batch_size	seed	config	mode	source	groups	banks	bigram_buckets	peak_vram_mb	val_bpb	num_steps	num_params_M	attnres_final_latest	attnres_final_x0	attnres_final_register	attnres_final_entropy	attnres_final_memory_total	attnres_final_memory_unigram	attnres_final_memory_bigram_bank0	attnres_final_memory_bigram_bank1	attnres_final_memory_bigram_bank2	attnres_final_memory_bigram_bank3	attnres_final_memory_query_gate	attnres_final_memory_query_norm	attnres_final_memory_query_delta_norm
EOF

parse_log() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5" seed="$6" name="$7" mode="$8" source="$9" groups="${10}" banks="${11}" buckets="${12}" log_path="${13}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps','num_params_M',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_total','attnres_final_memory_unigram','attnres_final_memory_bigram_bank0','attnres_final_memory_bigram_bank1','attnres_final_memory_bigram_bank2','attnres_final_memory_bigram_bank3',
  'attnres_final_memory_query_gate','attnres_final_memory_query_norm','attnres_final_memory_query_delta_norm',
]
row = [${size_tag@Q}, ${depth@Q}, ${head_dim@Q}, ${dbs@Q}, ${tbs@Q}, ${seed@Q}, ${name@Q}, ${mode@Q}, ${source@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_case() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5" seed="$6" name="$7" mode="$8" source="$9" groups="${10}" banks="${11}" buckets="${12}" extra_delta="${13}"
  local log_path="$LOG_DIR/${size_tag}_${name}_seed${seed}.log"
  echo "== ${size_tag} ${name} seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_DEPTH="$depth" \
  AUTORESEARCH_HEAD_DIM="$head_dim" \
  AUTORESEARCH_DEVICE_BATCH_SIZE="$dbs" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="$tbs" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="$mode" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="0.85" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_DELTA_SCALE="$extra_delta" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="1.0" \
  scripts/train_final_memory_static.sh
  parse_log "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" "$name" "$mode" "$source" "$groups" "$banks" "$buckets" "$log_path"
}

run_baseline() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5" seed="$6"
  local log_path="$LOG_DIR/${size_tag}_strong_baseline_seed${seed}.log"
  echo "== ${size_tag} strong_baseline seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_DEPTH="$depth" \
  AUTORESEARCH_HEAD_DIM="$head_dim" \
  AUTORESEARCH_DEVICE_BATCH_SIZE="$dbs" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="$tbs" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  scripts/train_attnres.sh
  parse_log "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" strong_baseline off off 0 0 0 "$log_path"
}

if [[ ! -f "$SELECTED" ]]; then
  echo "Missing $SELECTED. Run repro_model_size_probe.sh first." >&2
  exit 1
fi

while IFS=$'\t' read -r size_tag depth head_dim dbs tbs; do
  [[ "$size_tag" == "size_tag" ]] && continue
  for seed in 45 46; do
    run_baseline "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed"
    run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" blend_ngram_ref static tied_bigram 4 4 1048576 1.0
    run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" unified_ngram_projected unified_projected tied_bigram 1 4 1048576 1.0
    run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" query_bigram_only_projected query_projected tied_bigram_only 1 4 1048576 1.0
    run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$seed" query_ngram_projected query_projected tied_bigram 1 4 1048576 1.0
  done
done < "$SELECTED"
