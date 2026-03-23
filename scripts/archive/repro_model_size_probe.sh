#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/model_size_probe
RESULTS=results/raw/model_size_probe_results.tsv
LOG_DIR=runs/model_size_probe
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

cat > "$RESULTS" <<'EOF'
size_tag	depth	head_dim	device_batch_size	total_batch_size	config	mode	source	groups	banks	bigram_buckets	peak_vram_mb	val_bpb	num_steps	num_params_M	attnres_final_latest	attnres_final_x0	attnres_final_register	attnres_final_entropy	attnres_final_memory_total	attnres_final_memory_query_gate	attnres_final_memory_query_norm	attnres_final_memory_query_delta_norm
EOF

parse_log() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5" name="$6" mode="$7" source="$8" groups="$9" banks="${10}" buckets="${11}" log_path="${12}"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps','num_params_M',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_entropy',
  'attnres_final_memory_total','attnres_final_memory_query_gate','attnres_final_memory_query_norm','attnres_final_memory_query_delta_norm',
]
row = [${size_tag@Q}, ${depth@Q}, ${head_dim@Q}, ${dbs@Q}, ${tbs@Q}, ${name@Q}, ${mode@Q}, ${source@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_case() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5" name="$6" mode="$7" source="$8" groups="$9" banks="${10}" buckets="${11}" extra_delta="${12}"
  local log_path="$LOG_DIR/${size_tag}_${name}.log"
  echo "== ${size_tag} ${name} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="45" \
  AUTORESEARCH_DEPTH="$depth" \
  AUTORESEARCH_HEAD_DIM="$head_dim" \
  AUTORESEARCH_DEVICE_BATCH_SIZE="$dbs" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="$tbs" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=8 \
  AUTORESEARCH_TIME_BUDGET=180 \
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
  parse_log "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" "$name" "$mode" "$source" "$groups" "$banks" "$buckets" "$log_path"
}

run_baseline() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5"
  local log_path="$LOG_DIR/${size_tag}_strong_baseline.log"
  echo "== ${size_tag} strong_baseline =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="45" \
  AUTORESEARCH_DEPTH="$depth" \
  AUTORESEARCH_HEAD_DIM="$head_dim" \
  AUTORESEARCH_DEVICE_BATCH_SIZE="$dbs" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="$tbs" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=8 \
  AUTORESEARCH_TIME_BUDGET=180 \
  scripts/train_attnres.sh
  parse_log "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" strong_baseline off off 0 0 0 "$log_path"
}

run_size() {
  local size_tag="$1" depth="$2" head_dim="$3" dbs="$4" tbs="$5"
  run_baseline "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs"
  run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" blend_ngram_ref static tied_bigram 4 4 1048576 1.0
  run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" unified_ngram_projected unified_projected tied_bigram 1 4 1048576 1.0
  run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" query_bigram_only_projected query_projected tied_bigram_only 1 4 1048576 1.0
  run_case "$size_tag" "$depth" "$head_dim" "$dbs" "$tbs" query_ngram_projected query_projected tied_bigram 1 4 1048576 1.0
}

run_size s384_d6 6 64 64 16384
run_size m512_d8 8 64 16 16384
run_size l640_d10 10 64 8 16384
