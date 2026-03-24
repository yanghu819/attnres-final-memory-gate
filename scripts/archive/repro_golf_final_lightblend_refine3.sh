#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/golf_final_lightblend_refine3

RESULTS=results/raw/golf_final_lightblend_refine3_results.tsv
LOG_DIR=runs/golf_final_lightblend_refine3

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
Path("results/raw/golf_final_lightblend_refine3_results.tsv").write_text(
    "\t".join(
        [
            "config",
            "seed",
            "peak_vram_mb",
            "val_bpb",
            "num_steps",
            "attnres_final_memory_delta_norm",
            "attnres_final_memory_gate",
        ]
    )
    + "\n"
)
PY

parse_log() {
  local name="$1" seed="$2" log_path="$3"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  "peak_vram_mb", "val_bpb", "num_steps",
  "attnres_final_memory_delta_norm", "attnres_final_memory_gate",
]
row = [${name@Q}, ${seed@Q}]
row.extend(metrics.get(k, "") for k in keys)
print("\t".join(str(x) for x in row))
PY
}

run_case() {
  local name="$1" seed="$2" rank="$3" buckets="$4" cap="$5"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=64 \
  AUTORESEARCH_TIME_BUDGET=180 \
  AUTORESEARCH_DEPTH=10 \
  AUTORESEARCH_ASPECT_RATIO=8 \
  AUTORESEARCH_DEVICE_BATCH_SIZE=64 \
  AUTORESEARCH_TOTAL_BATCH_SIZE=16384 \
  AUTORESEARCH_ATTNRES_MODE=block \
  AUTORESEARCH_ATTNRES_BLOCK_SIZE=2 \
  AUTORESEARCH_ATTNRES_WEIGHT_MODE=softmax \
  AUTORESEARCH_ATTNRES_NUM_REGISTERS=0 \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTERS=0 \
  AUTORESEARCH_ATTNRES_REGISTER_SCOPE=all \
  AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE=off \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE=static \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE=tied_bigram_factorized \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE=rmsnorm \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE=0.25 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT=4.0 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK="$rank" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS=2 \
  AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-/root/autodl-tmp/.cache/autoresearch}" \
  AUTORESEARCH_BASE_URL="${AUTORESEARCH_BASE_URL:-https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main}" \
  AUTORESEARCH_MAX_SEQ_LEN="${AUTORESEARCH_MAX_SEQ_LEN:-256}" \
  AUTORESEARCH_EVAL_TOKENS="${AUTORESEARCH_EVAL_TOKENS:-1048576}" \
  AUTORESEARCH_WINDOW_PATTERN="${AUTORESEARCH_WINDOW_PATTERN:-L}" \
  AUTORESEARCH_ATTN_BACKEND="${AUTORESEARCH_ATTN_BACKEND:-sdpa}" \
  AUTORESEARCH_USE_COMPILE="${AUTORESEARCH_USE_COMPILE:-0}" \
  AUTORESEARCH_NANOGPT_REF="${AUTORESEARCH_NANOGPT_REF:-1}" \
  "$PYTHON_BIN" -m uv run python -m autoresearch_attnres_project.cli train --preset attnres_block2 \
    > "$log_path" 2>&1
  parse_log "$name" "$seed" "$log_path"
}

for seed in 45 46; do
  run_case "golf_final_lightblend_r64_b32768_c090" "$seed" 64 32768 0.90
  run_case "golf_final_lightblend_r96_b32768_c090" "$seed" 96 32768 0.90
  run_case "golf_final_lightblend_r64_b65536_c090" "$seed" 64 65536 0.90
  run_case "golf_final_lightblend_r96_b65536_c090" "$seed" 96 65536 0.90
  run_case "golf_final_lightblend_r64_b32768_c095" "$seed" 64 32768 0.95
done
