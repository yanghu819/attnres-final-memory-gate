#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/ablation
RESULTS=results/raw/ablation_matrix.tsv
LOG_DIR=runs/ablation
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/ablation_matrix.tsv').write_text('\t'.join([
    'config','family','seed','reference_lr','register_scope','token_register_bias_mode','token_register_scale','val_bpb','num_steps',
    'attnres_layer_latest','attnres_layer_x0','attnres_layer_register','attnres_layer_maxprob','attnres_layer_entropy',
    'attnres_final_latest','attnres_final_x0','attnres_final_register','attnres_final_maxprob','attnres_final_entropy','peak_vram_mb'
]) + '\n')
PY

parse_log() {
  local name="$1"
  local family="$2"
  local seed="$3"
  local lr="$4"
  local scope="$5"
  local bias_mode="$6"
  local scale="$7"
  local log_path="$8"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
name = ${name@Q}
family = ${family@Q}
seed = ${seed@Q}
lr = ${lr@Q}
scope = ${scope@Q}
bias_mode = ${bias_mode@Q}
scale = ${scale@Q}
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
fields = [
    name, family, seed, lr, scope, bias_mode, scale, metrics.get('val_bpb', ''), metrics.get('num_steps', ''),
    metrics.get('attnres_layer_latest', ''), metrics.get('attnres_layer_x0', ''), metrics.get('attnres_layer_register', ''), metrics.get('attnres_layer_maxprob', ''),
    metrics.get('attnres_layer_entropy', ''), metrics.get('attnres_final_latest', ''), metrics.get('attnres_final_x0', ''), metrics.get('attnres_final_register', ''),
    metrics.get('attnres_final_maxprob', ''), metrics.get('attnres_final_entropy', ''), metrics.get('peak_vram_mb', ''),
]
print('\t'.join(fields))
PY
}

run_family() {
  local family="$1"
  local seed="$2"
  local lr="$3"
  local scope="$4"
  local bias_mode="$5"
  local scale="$6"
  local script="$7"
  local name="ablation_${family}_reflr${lr//./p}_seed${seed}"
  local log_path="$LOG_DIR/${name}.log"
  local done_path="$log_path.done"
  echo "== $name =="
  rm -f "$done_path"
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="$lr" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_REGISTER_SCOPE="$scope" \
  AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE="$scale" \
  "$script"
  parse_log "$name" "$family" "$seed" "$lr" "$scope" "$bias_mode" "$scale" "$log_path"
  : > "$done_path"
}

for seed in 41 42 43; do
  run_family attnres_block2 "$seed" 0.0012 all static 0.0 scripts/train_attnres.sh
  run_family projected_deepemb_final_static "$seed" 0.0012 final_only static 0.3125 scripts/train_projected_static.sh
  run_family projected_deepemb_final "$seed" 0.0012 final_only projected 0.3125 scripts/train_projected.sh
  run_family projected_deepemb_all "$seed" 0.0012 all projected 0.25 scripts/train_projected_all.sh
done
