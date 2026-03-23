#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/lm_head_rotation
RESULTS=results/raw/lm_head_rotation_results.tsv
LOG_DIR=runs/lm_head_rotation
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/lm_head_rotation_results.tsv').write_text('\t'.join([
  'config','seed','untie_lm_head','rotate_every','rotate_rank','rotate_alpha',
  'peak_vram_mb','val_bpb','num_steps',
  'tie_lm_head','lm_head_rotation_updates','lm_head_rotation_lost_before','lm_head_rotation_lost_after',
  'attnres_final_latest','attnres_final_x0','attnres_final_entropy',
  'attnres_lm_head_grad_norm','attnres_wte_grad_norm'
]) + '\n')
PY

parse_log() {
  local name="$1" seed="$2" untie="$3" every="$4" rank="$5" alpha="$6" log_path="$7"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_summary_metrics
metrics = parse_summary_metrics(Path(${log_path@Q}))
keys = [
  'peak_vram_mb','val_bpb','num_steps',
  'tie_lm_head','lm_head_rotation_updates','lm_head_rotation_lost_before','lm_head_rotation_lost_after',
  'attnres_final_latest','attnres_final_x0','attnres_final_entropy',
  'attnres_lm_head_grad_norm','attnres_wte_grad_norm',
]
row = [${name@Q}, ${seed@Q}, ${untie@Q}, ${every@Q}, ${rank@Q}, ${alpha@Q}]
row.extend(metrics.get(k, '') for k in keys)
print('\t'.join(str(x) for x in row))
PY
}

run_case() {
  local name="$1" seed="$2" untie="$3" every="$4" rank="$5" alpha="$6"
  local log_path="$LOG_DIR/${name}_seed${seed}.log"
  echo "== ${name}_seed${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_DEVICE_BATCH_SIZE="${AUTORESEARCH_DEVICE_BATCH_SIZE:-32}" \
  AUTORESEARCH_TOTAL_BATCH_SIZE="${AUTORESEARCH_TOTAL_BATCH_SIZE:-16384}" \
  AUTORESEARCH_UNTIE_LM_HEAD="$untie" \
  AUTORESEARCH_LM_HEAD_ROTATE_EVERY="$every" \
  AUTORESEARCH_LM_HEAD_ROTATE_RANK="$rank" \
  AUTORESEARCH_LM_HEAD_ROTATE_ALPHA="$alpha" \
  AUTORESEARCH_LM_HEAD_ROTATE_MAX_POSITIONS="${AUTORESEARCH_LM_HEAD_ROTATE_MAX_POSITIONS:-64}" \
  AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_ROWS="${AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_ROWS:-2048}" \
  AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_DEVICE="${AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_DEVICE:-cpu}" \
  scripts/train_attnres.sh
  parse_log "$name" "$seed" "$untie" "$every" "$rank" "$alpha" "$log_path"
}

for seed in 45 46 47 48; do
  run_case strong_baseline_tied "$seed" 0 0 0 0.0
  run_case strong_baseline_untied "$seed" 1 0 0 0.0
  run_case lmrotate_r8_a005 "$seed" 1 64 8 0.05
  run_case lmrotate_r16_a005 "$seed" 1 64 16 0.05
  run_case lmrotate_r16_a010 "$seed" 1 64 16 0.10
done
