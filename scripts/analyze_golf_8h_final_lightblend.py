from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw" / "golf_8h_final_lightblend_results.tsv"
OUT = ROOT / "results" / "golf_8h_final_lightblend_analysis.txt"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    rows = []
    with RAW.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["config"])].append(row)

    lines = []
    for stage in sorted({k[0] for k in grouped}):
        lines.append(f"[{stage}]")
        stage_items = []
        for (s, cfg), cfg_rows in grouped.items():
            if s != stage:
                continue
            vals = [float(r["val_bpb"]) for r in cfg_rows if r["val_bpb"]]
            vrams = [float(r["peak_vram_mb"]) for r in cfg_rows if r["peak_vram_mb"]]
            if not vals:
                continue
            stage_items.append((mean(vals), cfg, std(vals), mean(vrams), len(vals)))
        for m, cfg, sd, vram, n in sorted(stage_items):
            lines.append(f"{cfg}\tmean={m:.6f}\tstd={sd:.6f}\tmean_vram_mb={vram:.1f}\tn={n}")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
