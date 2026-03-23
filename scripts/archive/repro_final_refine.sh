#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/final_refine
RESULTS=results/raw/final_refine_results.tsv
LOG_DIR=runs/final_refine
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/final_refine_results.tsv').write_text('\t'.join([
    'config','seed','reference_lr','scale','scale_start','scale_warmup_steps','delta_scale','weight_cap',
    'val_bpb','num_steps','attnres_layer_latest','attnres_layer_x0','attnres_layer_register',
    'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_aux_loss','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1"
  local seed="$2"
  local lr="$3"
  local scale="$4"
  local scale_start="$5"
  local scale_warmup="$6"
  local delta_scale="$7"
  local weight_cap="$8"
  local log_path="$9"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
name = ${name@Q}
seed = ${seed@Q}
lr = ${lr@Q}
scale = ${scale@Q}
scale_start = ${scale_start@Q}
scale_warmup = ${scale_warmup@Q}
delta_scale = ${delta_scale@Q}
weight_cap = ${weight_cap@Q}
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
fields = [
    name, seed, lr, scale, scale_start, scale_warmup, delta_scale, weight_cap,
    metrics.get('val_bpb', ''), metrics.get('num_steps', ''),
    metrics.get('attnres_layer_latest', ''), metrics.get('attnres_layer_x0', ''), metrics.get('attnres_layer_register', ''),
    metrics.get('attnres_final_latest', ''), metrics.get('attnres_final_x0', ''), metrics.get('attnres_final_register', ''),
    metrics.get('attnres_aux_loss', ''), metrics.get('peak_vram_mb', ''),
]
print('\t'.join(fields))
PY
}

run_variant() {
  local name="$1"
  local seed="$2"
  local scale="$3"
  local scale_start="$4"
  local scale_warmup="$5"
  local delta_scale="$6"
  local weight_cap="$7"
  local lr="0.0012"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_REGISTER_SCOPE="final_only" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE="$scale" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_START="$scale_start" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS="$scale_warmup" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_DELTA_SCALE="$delta_scale" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_WEIGHT_CAP="$weight_cap" \
  scripts/train_projected.sh
  parse_log "$name" "$seed" "$lr" "$scale" "$scale_start" "$scale_warmup" "$delta_scale" "$weight_cap" "$log_path"
}

for seed in 45 46 47 48; do
  run_variant control_final         "$seed" 0.3125 0.3125 0   1.0 0.0
  run_variant late_onset_final      "$seed" 0.3125 0.0    64  1.0 0.0
  run_variant delta_half_final      "$seed" 0.3125 0.3125 0   0.5 0.0
  run_variant capped_final          "$seed" 0.3125 0.3125 0   1.0 0.9
done

