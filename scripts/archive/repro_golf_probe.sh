#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/golf_probe
RESULTS=results/raw/golf_probe_results.tsv
LOG_DIR=runs/golf_probe
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
Path('results/raw/golf_probe_results.tsv').write_text('\t'.join([
    'config','seed','peak_vram_mb','val_bpb','num_steps','attnres_input_memory_delta_norm','attnres_input_memory_smear_gate','attnres_final_memory_delta_norm','attnres_final_memory_gate'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" log_path="$3"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'attnres_input_memory_delta_norm','attnres_input_memory_smear_gate',
  'attnres_final_memory_delta_norm','attnres_final_memory_gate',
]
row = [${name@Q}, ${seed@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_preset() {
  local preset="$1" seed="$2"
  local log_path="$LOG_DIR/${preset}_seed${seed}.log"
  echo "== ${preset}_seed${seed} =="
  export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-/root/autodl-tmp/.cache/autoresearch}"
  export AUTORESEARCH_BASE_URL="${AUTORESEARCH_BASE_URL:-https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main}"
  export AUTORESEARCH_MAX_SEQ_LEN="${AUTORESEARCH_MAX_SEQ_LEN:-256}"
  export AUTORESEARCH_EVAL_TOKENS="${AUTORESEARCH_EVAL_TOKENS:-1048576}"
  export AUTORESEARCH_WINDOW_PATTERN="${AUTORESEARCH_WINDOW_PATTERN:-L}"
  export AUTORESEARCH_ATTN_BACKEND="${AUTORESEARCH_ATTN_BACKEND:-sdpa}"
  export AUTORESEARCH_USE_COMPILE="${AUTORESEARCH_USE_COMPILE:-0}"
  export AUTORESEARCH_NANOGPT_REF="${AUTORESEARCH_NANOGPT_REF:-1}"
  if [[ "$preset" == "golf_input_bigram_smear_tiny" ]]; then
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
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE=bigram \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE=0.25 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM=16 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS=8192 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS=2 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE=1.0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE=0.0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR=1 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT=-4.0 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE=off \
    "$PYTHON_BIN" -m uv run python -m autoresearch_attnres_project.cli train --preset attnres_block2 > "$log_path" 2>&1
  elif [[ "$preset" == "golf_final_lightblend_tiny" ]]; then
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
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP=0.85 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT=4.0 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK=16 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS=8192 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS=2 \
    "$PYTHON_BIN" -m uv run python -m autoresearch_attnres_project.cli train --preset attnres_block2 > "$log_path" 2>&1
  else
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
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE=off \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE=full \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS=0 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS=1 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK=0 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP=0.0 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE=1.0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM=16 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS=0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS=1 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE=1.0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE=1.0 \
    AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR=0 \
    "$PYTHON_BIN" -m uv run python -m autoresearch_attnres_project.cli train --preset attnres_block2 > "$log_path" 2>&1
  fi
  parse_log "$preset" "$seed" "$log_path"
}

seed=45
run_preset golf_attnres_tiny "$seed"
run_preset golf_input_bigram_smear_tiny "$seed"
run_preset golf_final_lightblend_tiny "$seed"
