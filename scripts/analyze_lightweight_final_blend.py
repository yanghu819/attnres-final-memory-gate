from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

root = Path(__file__).resolve().parent.parent
raw_path = root / "results" / "raw" / "lightweight_final_blend_results.tsv"
out_path = root / "results" / "lightweight_final_blend_analysis.txt"

rows = []
with raw_path.open() as f:
    rows.extend(csv.DictReader(f, delimiter="\t"))

by_cfg = defaultdict(list)
for row in rows:
    try:
        row["val_bpb"] = float(row["val_bpb"])
        row["peak_vram_mb"] = float(row["peak_vram_mb"])
    except Exception:
        continue
    by_cfg[row["config"]].append(row)

baseline_mean = None
if "strong_baseline" in by_cfg:
    baseline_mean = mean(r["val_bpb"] for r in by_cfg["strong_baseline"])

lines = []
for cfg, cfg_rows in sorted(by_cfg.items(), key=lambda kv: mean(r["val_bpb"] for r in kv[1])):
    vals = [r["val_bpb"] for r in cfg_rows]
    vrams = [r["peak_vram_mb"] for r in cfg_rows]
    line = (
        f"{cfg}\tmean={mean(vals):.6f}\tstd={pstdev(vals) if len(vals) > 1 else 0.0:.6f}\t"
        f"mean_vram_mb={mean(vrams):.1f}\tn={len(vals)}"
    )
    if baseline_mean is not None:
        line += f"\tdelta_vs_baseline={mean(vals) - baseline_mean:+.6f}"
    lines.append(line)

out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
print(out_path)
