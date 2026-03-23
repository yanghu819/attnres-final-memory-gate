#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/lightweight_final_blend results/figs
RESULTS=results/raw/lightweight_final_blend_results.tsv
LOG_DIR=runs/lightweight_final_blend
if [ -x /root/miniconda3/bin/python ]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path("results/raw/lightweight_final_blend_results.tsv").write_text("\t".join([
    "config","seed","source","rank","bigram_buckets","bigram_banks","peak_vram_mb","val_bpb","num_steps",
    "attnres_final_latest","attnres_final_x0","attnres_final_register","attnres_final_entropy",
    "attnres_final_memory_gate","attnres_final_memory_delta_norm","attnres_final_memory_rms",
    "attnres_final_memory_embed_grad_norm","attnres_final_memory_proj_grad_norm"
]) + "\n")
PY

parse_log() {
  local name="$1" seed="$2" source="$3" rank="$4" buckets="$5" banks="$6" log_path="$7"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  "peak_vram_mb","val_bpb","num_steps",
  "attnres_final_latest","attnres_final_x0","attnres_final_register","attnres_final_entropy",
  "attnres_final_memory_gate","attnres_final_memory_delta_norm","attnres_final_memory_rms",
  "attnres_final_memory_embed_grad_norm","attnres_final_memory_proj_grad_norm",
]
row = [${name@Q}, ${seed@Q}, ${source@Q}, ${rank@Q}, ${buckets@Q}, ${banks@Q}]
row.extend(metrics.get(k, "") for k in keys)
print("\t".join(str(x) for x in row))
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
  AUTORESEARCH_DEVICE_BATCH_SIZE=32 \
  scripts/train_attnres.sh
  parse_log "strong_baseline" "$seed" off 0 0 0 "$log_path"
}

run_blend() {
  local name="$1" seed="$2" source="$3" rank="$4" buckets="$5" banks="$6"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_DEVICE_BATCH_SIZE=32 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE="static" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="$source" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE="rmsnorm" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="0.25" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="0.85" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK="$rank" \
  scripts/train_attnres.sh
  parse_log "$name" "$seed" "$source" "$rank" "$buckets" "$banks" "$log_path"
}

for seed in 45 46 47 48; do
  run_control "$seed"
  run_blend blend_ngram_ref "$seed" tied_bigram 0 1048576 4
  run_blend lightblend_r32_b64k "$seed" tied_bigram_factorized 32 65536 4
  run_blend lightblend_r64_b64k "$seed" tied_bigram_factorized 64 65536 4
  run_blend lightblend_r64_b256k "$seed" tied_bigram_factorized 64 262144 4
  run_blend lightblend_r96_b256k "$seed" tied_bigram_factorized 96 262144 4
done

PYTHONPATH=src "$PYTHON_BIN" scripts/plot_lightweight_final_blend.py
PYTHONPATH=src "$PYTHON_BIN" scripts/analyze_lightweight_final_blend.py
