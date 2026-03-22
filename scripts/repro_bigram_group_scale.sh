#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw runs/bigram_group_scale
RESULTS=results/raw/bigram_group_scale_results.tsv
LOG_DIR=runs/bigram_group_scale
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi
PROBE_RESULTS=${PROBE_RESULTS:-results/raw/bigram_vram_probe_results.tsv}

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
Path('results/raw/bigram_group_scale_results.tsv').write_text('\t'.join([
  'config','groups','banks','bigram_buckets','value_scale','gate_cap','seed','peak_vram_mb','val_bpb','num_steps',
  'attnres_final_memory_gate','attnres_final_memory_gate_std','attnres_final_memory_delta_norm'
]) + '\n')
PY

select_configs() {
  PROBE_RESULTS="$PROBE_RESULTS" "$PYTHON_BIN" - <<'PY'
import csv, os
from pathlib import Path
path = Path(os.environ['PROBE_RESULTS'])
rows=[]
with path.open() as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['status'] == 'ok' and row['peak_vram_mb']:
            rows.append(row)
if not rows:
    raise SystemExit('no successful probe rows')
target=float(rows[0]['target_vram_mb'])
# prefer configs below target and closest to it, with distinct (groups,banks,buckets)
rows.sort(key=lambda r: (abs(target - float(r['peak_vram_mb'])), -float(r['peak_vram_mb'])))
seen=set()
chosen=[]
for row in rows:
    key=(row['groups'], row['banks'], row['bigram_buckets'])
    if key in seen:
        continue
    seen.add(key)
    chosen.append(row)
    if len(chosen) == 4:
        break
for row in chosen:
    print('\t'.join([row['groups'], row['banks'], row['bigram_buckets']]))
PY
}

parse_log() {
  local name="$1" groups="$2" banks="$3" buckets="$4" scale="$5" cap="$6" seed="$7" log_path="$8"
  PYTHONPATH=src "$PYTHON_BIN" - <<PY >> "$RESULTS"
from pathlib import Path
from autoresearch_attnres_project.metrics import parse_required_summary_metrics
metrics = parse_required_summary_metrics(Path(${log_path@Q}))
print('\t'.join([
  ${name@Q}, ${groups@Q}, ${banks@Q}, ${buckets@Q}, ${scale@Q}, ${cap@Q}, ${seed@Q},
  str(metrics.get('peak_vram_mb','')),
  str(metrics.get('val_bpb','')),
  str(metrics.get('num_steps','')),
  str(metrics.get('attnres_final_memory_gate','')),
  str(metrics.get('attnres_final_memory_gate_std','')),
  str(metrics.get('attnres_final_memory_delta_norm','')),
]))
PY
}

run_cfg() {
  local name="$1" groups="$2" banks="$3" buckets="$4" scale="$5" cap="$6" seed="$7"
  local log_path="$LOG_DIR/${name}_g${groups}_bk${banks}_b${buckets}_seed${seed}.log"
  echo "== train ${name} g=${groups} banks=${banks} buckets=${buckets} seed=${seed} =="
  AUTORESEARCH_LOG_PATH="$log_path" \
  AUTORESEARCH_SEED="$seed" \
  AUTORESEARCH_REFERENCE_LR="0.0012" \
  AUTORESEARCH_MAX_STEPS=264 \
  AUTORESEARCH_TIME_BUDGET=600 \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE="tied_bigram" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GROUPS="$groups" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS="$banks" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS="$buckets" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE="$scale" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP="$cap" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT="4.0" \
  AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_SCALE="1.0" \
  scripts/train_final_memory_static.sh
  parse_log "$name" "$groups" "$banks" "$buckets" "$scale" "$cap" "$seed" "$log_path"
}

# include current best references
for seed in 45 46 47 48; do
  run_cfg current_ref_1g_1b_16k 1 1 16384 0.25 0.85 "$seed"
done

while IFS=$'\t' read -r groups banks buckets; do
  [[ -z "$groups" ]] && continue
  name="bigram_g${groups}_bk${banks}_b${buckets}"
  for seed in 45 46 47 48; do
    run_cfg "$name" "$groups" "$banks" "$buckets" 0.25 0.85 "$seed"
  done
done < <(select_configs)
