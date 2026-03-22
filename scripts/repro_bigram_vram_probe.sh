#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/bigram_vram_probe
RESULTS=results/raw/bigram_vram_probe_results.tsv
LOG_DIR=runs/bigram_vram_probe
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

GPU_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
GPU_TARGET_MB=$(( GPU_TOTAL_MB * 80 / 100 ))

echo "gpu_total_mb=${GPU_TOTAL_MB}"
echo "gpu_target_mb=${GPU_TARGET_MB}"

"$PYTHON_BIN" - <<PY
from pathlib import Path
Path(${RESULTS@Q}).write_text('\t'.join([
  'config','groups','banks','bigram_buckets','value_scale','gate_cap','seed','status',
  'target_vram_mb','peak_vram_mb','val_bpb','num_steps','attnres_final_memory_gate','attnres_final_memory_delta_norm'
]) + '\n')
PY

parse_success() {
  local name="$1" groups="$2" banks="$3" buckets="$4" scale="$5" cap="$6" seed="$7" log_path="$8"
  local target_mb="$9"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${cap@Q}, ${seed@Q}, 'ok',
  ${target_mb@Q},
  str(metrics.get('peak_vram_mb','')),
  str(metrics.get('val_bpb','')),
  str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_memory_gate','')),
  str(metrics.get('attnres_final_memory_delta_norm','')),
]))
PY
}

record_fail() {
  local name="$1" groups="$2" banks="$3" buckets="$4" scale="$5" cap="$6" seed="$7" target_mb="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tfail\t%s\t\t\t\t\t\n' \
    "$name" "$groups" "$banks" "$buckets" "$scale" "$cap" "$seed" "$target_mb" >> "$RESULTS"
}

run_probe() {
  local name="$1" groups="$2" banks="$3" buckets="$4" scale="$5" cap="$6" seed="$7"
  local log_path="$LOG_DIR/${name}_g${groups}_bk${banks}_b${buckets}_seed${seed}.log"
  echo "== probe ${name} g=${groups} banks=${banks} buckets=${buckets} =="
  set +e
  timeout 420s env \
    AUTORESEARCH_LOG_PATH="$log_path" \
    AUTORESEARCH_SEED="$seed" \
    AUTORESEARCH_REFERENCE_LR="0.0012" \
    AUTORESEARCH_MAX_STEPS=8 \
    AUTORESEARCH_TIME_BUDGET=180 \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$scale" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$cap" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
    AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="1.0" \
    scripts/train_final_memory_static.sh
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    parse_success "$name" "$groups" "$banks" "$buckets" "$scale" "$cap" "$seed" "$log_path" "$GPU_TARGET_MB"
  else
    record_fail "$name" "$groups" "$banks" "$buckets" "$scale" "$cap" "$seed" "$GPU_TARGET_MB"
  fi
}

SEED=45
# Size ladder first. Groups don't materially change VRAM; use g=1 for sizing.
run_probe size_1x16k   1 1 16384   0.25 0.85 "$SEED"
run_probe size_1x64k   1 1 65536   0.25 0.85 "$SEED"
run_probe size_2x64k   1 2 65536   0.25 0.85 "$SEED"
run_probe size_4x64k   1 4 65536   0.25 0.85 "$SEED"
run_probe size_8x64k   1 8 65536   0.25 0.85 "$SEED"
run_probe size_4x256k  1 4 262144  0.25 0.85 "$SEED"
run_probe size_8x256k  1 8 262144  0.25 0.85 "$SEED"
run_probe size_4x1m    1 4 1048576 0.25 0.85 "$SEED"
run_probe size_8x1m    1 8 1048576 0.25 0.85 "$SEED"

# Group sensitivity near the larger safe sizes.
run_probe groups_2_8x256k 2 8 262144  0.25 0.85 "$SEED"
run_probe groups_4_8x256k 4 8 262144  0.25 0.85 "$SEED"
run_probe groups_8_8x256k 8 8 262144  0.25 0.85 "$SEED"
run_probe groups_2_4x1m   2 4 1048576 0.25 0.85 "$SEED"
run_probe groups_4_4x1m   4 4 1048576 0.25 0.85 "$SEED"
run_probe groups_8_4x1m   8 4 1048576 0.25 0.85 "$SEED"
