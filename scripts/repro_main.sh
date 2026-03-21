#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/main
RESULTS=results/raw/main_fair_matrix.tsv
LOG_DIR=runs/main
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/main_fair_matrix.tsv').write_text('\t'.join([
    'config','family','seed','reference_lr','val_bpb','num_steps','budget_mode','attnres_mode','attnres_block_size','attnres_weight_mode',
    'attnres_num_registers','attnres_token_registers','attnres_register_scope','attnres_token_register_mode','attnres_token_register_bias_mode',
    'attnres_token_register_value_mode','attnres_token_register_scale','attnres_layer_latest','attnres_layer_x0','attnres_layer_register',
    'attnres_layer_maxprob','attnres_layer_entropy','attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_maxprob',
    'attnres_final_entropy','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1"
  local family="$2"
  local seed="$3"
  local lr="$4"
  local log_path="$5"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
name = ${name@Q}
family = ${family@Q}
seed = ${seed@Q}
lr = ${lr@Q}
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
fields = [
    name, family, seed, lr,
    metrics.get('val_bpb', ''), metrics.get('num_steps', ''), metrics.get('budget_mode', ''), metrics.get('attnres_mode', ''),
    metrics.get('attnres_block_size', ''), metrics.get('attnres_weight_mode', ''), metrics.get('attnres_num_registers', ''),
    metrics.get('attnres_token_registers', ''), metrics.get('attnres_register_scope', ''), metrics.get('attnres_token_register_mode', ''),
    metrics.get('attnres_token_register_bias_mode', ''), metrics.get('attnres_token_register_value_mode', ''), metrics.get('attnres_token_register_scale', ''),
    metrics.get('attnres_layer_latest', ''), metrics.get('attnres_layer_x0', ''), metrics.get('attnres_layer_register', ''), metrics.get('attnres_layer_maxprob', ''),
    metrics.get('attnres_layer_entropy', ''), metrics.get('attnres_final_latest', ''), metrics.get('attnres_final_x0', ''),
    metrics.get('attnres_final_register', ''), metrics.get('attnres_final_maxprob', ''), metrics.get('attnres_final_entropy', ''), metrics.get('peak_vram_mb', ''),
]
print('\t'.join(fields))
PY
}

run_family() {
  local family="$1"
  local seed="$2"
  local lr="$3"
  local script="$4"
  local name="main_${family}_reflr${lr//./p}_seed${seed}"
  local log_path="$LOG_DIR/${name}.log"
  local done_path="$log_path.done"
  echo "== $name =="
  rm -f "$done_path"
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  "$script"
  parse_log "$name" "$family" "$seed" "$lr" "$log_path"
  : > "$done_path"
}

for lr in 0.0003 0.0006 0.0010 0.0012; do
  for seed in 41 42 43; do
    run_family baseline_off "$seed" "$lr" scripts/train_baseline.sh
    run_family attnres_block2 "$seed" "$lr" scripts/train_attnres.sh
    run_family projected_deepemb_final "$seed" "$lr" scripts/train_projected.sh
  done
done
