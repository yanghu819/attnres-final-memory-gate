#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs/capped_refine
RESULTS=results/raw/capped_refine_results.tsv
LOG_DIR=runs/capped_refine
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/capped_refine_results.tsv').write_text('\t'.join([
  'config','seed','reference_lr','scale','delta_scale','weight_cap','val_bpb','num_steps',
  'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_aux_loss','peak_vram_mb'
]) + '\n')
PY
parse_log() {
  local name="$1" seed="$2" lr="$3" scale="$4" delta_scale="$5" weight_cap="$6" log_path="$7"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${seed@Q}, ${lr@Q}, ${scale@Q}, ${delta_scale@Q}, ${weight_cap@Q},
  str(metrics.get('val_bpb','')), str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_latest','')), str(metrics.get('attnres_final_x0','')),
  str(metrics.get('attnres_final_register','')), str(metrics.get('attnres_aux_loss','')),
  str(metrics.get('peak_vram_mb','')),
]))
PY
}
run_variant() {
  local name="$1" seed="$2" scale="$3" delta_scale="$4" weight_cap="$5"
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
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_START="$scale" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE_WARMUP_STEPS=0 \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_DELTA_SCALE="$delta_scale" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_WEIGHT_CAP="$weight_cap" \
  scripts/train_projected.sh
  parse_log "$name" "$seed" "$lr" "$scale" "$delta_scale" "$weight_cap" "$log_path"
}
for seed in 45 46 47 48; do
  run_variant control_cap00 "$seed" 0.3125 1.0 0.0
  run_variant cap085       "$seed" 0.3125 1.0 0.85
  run_variant cap090       "$seed" 0.3125 1.0 0.90
  run_variant cap095       "$seed" 0.3125 1.0 0.95
  run_variant cap090_d05   "$seed" 0.3125 0.5 0.90
done
