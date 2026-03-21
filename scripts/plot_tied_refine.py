#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_attnres_project.metrics import parse_log_curve

RESULTS = ROOT / "results" / "raw" / "tied_refine_results.tsv"
RUNS = ROOT / "runs" / "tied_refine"
OUT = ROOT / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "static_full_s025_c085": ("Full s=0.25 c=0.85", "#d1495b"),
    "tied_s020_c080": ("Tied s=0.20 c=0.80", "#1f3b4d"),
    "tied_s020_c085": ("Tied s=0.20 c=0.85", "#00798c"),
    "tied_s025_c080": ("Tied s=0.25 c=0.80", "#6a4c93"),
    "tied_s025_c085": ("Tied s=0.25 c=0.85", "#edae49"),
    "tied_s025_c090": ("Tied s=0.25 c=0.90", "#8f2d56"),
    "tied_s030_c085": ("Tied s=0.30 c=0.85", "#2a9d8f"),
    "tied_s03125_c085": ("Tied s=0.3125 c=0.85", "#f4a261"),
}


def load_group(df: pd.DataFrame, family: str):
    sub = df[df["config"] == family].copy().sort_values("seed")
    if sub.empty:
        return sub, []
    sub["val_bpb"] = sub["val_bpb"].astype(float)
    return sub, [RUNS / f"{name}_seed{seed}.log" for name, seed in zip(sub["config"], sub["seed"])]


def plot_curves(loaded):
    fig, ax = plt.subplots(figsize=(11.4, 6.6), dpi=180)
    for family, (label, color) in FAMILIES.items():
        df, logs = loaded[family]
        if df.empty:
            continue
        merged = None
        for i, log_path in enumerate(logs):
            if not log_path.exists():
                continue
            run = parse_log_curve(log_path)[["step", "loss_ema"]].rename(columns={"loss_ema": f"run_{i}"})
            merged = run if merged is None else merged.merge(run, on="step", how="inner")
        if merged is None or merged.empty:
            continue
        values = merged.drop(columns=["step"])
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=0).fillna(0.0)
        ax.plot(merged["step"], mean, color=color, linewidth=2.0, label=f"{label} (n={len(df)})")
        ax.fill_between(merged["step"], mean - std, mean + std, color=color, alpha=0.12)
    ax.set_title("Tied Final Memory Refinement")
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA train loss")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tied_refine_loss_curves.png")
    plt.close(fig)


def plot_val(loaded):
    order = [k for k in FAMILIES if not loaded[k][0].empty]
    fig, ax = plt.subplots(figsize=(11.4, 5.8), dpi=180)
    means, stds = [], []
    for family in order:
        df, _ = loaded[family]
        values = df["val_bpb"].tolist()
        means.append(sum(values) / len(values))
        stds.append(pd.Series(values).std(ddof=0))
    xs = range(len(order))
    ax.bar(xs, means, yerr=stds, color=[FAMILIES[k][1] for k in order], alpha=0.88, capsize=4)
    for i, family in enumerate(order):
        df, _ = loaded[family]
        offsets = [-0.12, -0.04, 0.04, 0.12][: len(df)]
        for xoff, y in zip(offsets, df["val_bpb"].tolist()):
            ax.scatter(i + xoff, y, color="#111111", s=18, alpha=0.75, zorder=3)
    ax.set_xticks(list(xs), [FAMILIES[k][0] for k in order], rotation=24, ha="right")
    ax.set_ylabel("val_bpb")
    ax.set_title("Tied Final Memory Refinement: val_bpb")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tied_refine_valbpb.png")
    plt.close(fig)


def export_summary(loaded):
    lines = ["config\tmean_val_bpb\tstd_val_bpb\tn"]
    for family in FAMILIES:
        df, _ = loaded[family]
        if df.empty:
            continue
        values = df["val_bpb"].tolist()
        mean = sum(values) / len(values)
        std = pd.Series(values).std(ddof=0)
        lines.append(f"{family}\t{mean:.6f}\t{std:.6f}\t{len(values)}")
    (ROOT / "results" / "tied_refine_results.tsv").write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    loaded = {family: load_group(df, family) for family in FAMILIES}
    plot_curves(loaded)
    plot_val(loaded)
    export_summary(loaded)


if __name__ == "__main__":
    main()
