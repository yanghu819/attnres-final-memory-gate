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

RESULTS = ROOT / "results" / "raw" / "static_gate_refine_results.tsv"
RUNS = ROOT / "runs" / "static_gate_refine"
OUT = ROOT / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "strong_baseline": ("AttnRes Block2", "#1f3b4d"),
    "projected_cap085": ("Old Projected + Cap 0.85", "#00798c"),
    "static_c080_s03125": ("Static cap=0.80 scale=0.3125", "#5c4b51"),
    "static_c085_s02500": ("Static cap=0.85 scale=0.25", "#8f2d56"),
    "static_c085_s03125": ("Static cap=0.85 scale=0.3125", "#d1495b"),
    "static_c085_s03750": ("Static cap=0.85 scale=0.375", "#edae49"),
    "static_c090_s03125": ("Static cap=0.90 scale=0.3125", "#6a4c93"),
}


def load_group(df: pd.DataFrame, family: str):
    sub = df[df["config"] == family].copy().sort_values("seed")
    if sub.empty:
        return sub, []
    sub["val_bpb"] = sub["val_bpb"].astype(float)
    return sub, [RUNS / f"{name}_seed{seed}.log" for name, seed in zip(sub["config"], sub["seed"])]


def plot_curves(loaded):
    fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=180)
    for family, (label, color) in GROUPS.items():
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
        ax.plot(merged["step"], mean, color=color, linewidth=2.1, label=f"{label} (n={len(df)})")
        ax.fill_between(merged["step"], mean - std, mean + std, color=color, alpha=0.14)
    ax.set_title("Static Final Memory Gate Ablation")
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA train loss")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_static_gate_refine_loss_curves.png")
    plt.close(fig)


def plot_val(loaded):
    order = [k for k in GROUPS if not loaded[k][0].empty]
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=180)
    means, stds = [], []
    for family in order:
        df, _ = loaded[family]
        values = df["val_bpb"].tolist()
        means.append(sum(values) / len(values))
        stds.append(pd.Series(values).std(ddof=0))
    xs = range(len(order))
    ax.bar(xs, means, yerr=stds, color=[GROUPS[k][1] for k in order], alpha=0.88, capsize=4)
    for i, family in enumerate(order):
        df, _ = loaded[family]
        offsets = [-0.12, -0.04, 0.04, 0.12][: len(df)]
        for xoff, y in zip(offsets, df["val_bpb"].tolist()):
            ax.scatter(i + xoff, y, color="#111111", s=18, alpha=0.75, zorder=3)
    ax.set_xticks(list(xs), [GROUPS[k][0] for k in order], rotation=20, ha="right")
    ax.set_ylabel("val_bpb")
    ax.set_title("Static Final Memory Gate: val_bpb")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_static_gate_refine_valbpb.png")
    plt.close(fig)


def export_summary(loaded):
    lines = ["config\tmean_val_bpb\tstd_val_bpb\tn"]
    for family in GROUPS:
        df, _ = loaded[family]
        if df.empty:
            continue
        values = df["val_bpb"].tolist()
        mean = sum(values) / len(values)
        std = pd.Series(values).std(ddof=0)
        lines.append(f"{family}\t{mean:.6f}\t{std:.6f}\t{len(values)}")
    (ROOT / "results" / "static_gate_refine_results.tsv").write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    loaded = {family: load_group(df, family) for family in GROUPS}
    plot_curves(loaded)
    plot_val(loaded)
    export_summary(loaded)


if __name__ == "__main__":
    main()
