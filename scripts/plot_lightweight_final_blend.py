from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
raw_path = root / "results" / "raw" / "lightweight_final_blend_results.tsv"
out_tsv = root / "results" / "lightweight_final_blend_results.tsv"
fig_path = root / "results" / "figs" / "fig_lightweight_final_blend_valbpb.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
with raw_path.open() as f:
    rows.extend(csv.DictReader(f, delimiter="\t"))

by_cfg = defaultdict(list)
for row in rows:
    try:
        by_cfg[row["config"]].append(float(row["val_bpb"]))
    except Exception:
        pass

items = sorted(((cfg, sum(vals) / len(vals)) for cfg, vals in by_cfg.items() if vals), key=lambda kv: kv[1])
with out_tsv.open("w") as f:
    f.write("config\tmean_val_bpb\n")
    for cfg, m in items:
        f.write(f"{cfg}\t{m:.6f}\n")

plt.figure(figsize=(10, 5))
plt.bar([cfg for cfg, _ in items], [m for _, m in items])
plt.xticks(rotation=35, ha="right")
plt.ylabel("val_bpb")
plt.tight_layout()
plt.savefig(fig_path, dpi=180)
print(fig_path)
