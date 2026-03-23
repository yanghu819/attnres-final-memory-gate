from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

root = Path(__file__).resolve().parent.parent
raw_path = root / 'results' / 'raw' / 'input_memory_refine_results.tsv'
out_path = root / 'results' / 'input_memory_refine_analysis.txt'

rows = []
with raw_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows.extend(reader)

by_cfg = defaultdict(list)
for row in rows:
    try:
        row['val_bpb'] = float(row['val_bpb'])
    except Exception:
        continue
    by_cfg[row['config']].append(row)

lines = []
control = None
if 'strong_baseline' in by_cfg:
    control = mean(r['val_bpb'] for r in by_cfg['strong_baseline'])
for cfg, cfg_rows in sorted(by_cfg.items(), key=lambda kv: mean(r['val_bpb'] for r in kv[1])):
    vals = [r['val_bpb'] for r in cfg_rows]
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    delta = '' if control is None else f"\tdelta_vs_baseline={m-control:+.6f}"
    lines.append(f"{cfg}\tmean={m:.6f}\tstd={s:.6f}\tn={len(vals)}{delta}")

out_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
print(out_path)
