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

RESULTS = ROOT / "results" / "raw" / "main_fair_matrix.tsv"
RUNS = ROOT / "runs" / "main"
OUT = ROOT / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "baseline_off": ("Autoresearch Baseline", "#222222"),
    "attnres_block2": ("AttnRes Block2 Strong Baseline", "#00798c"),
    "projected_deepemb_final": ("Projected Token Memory (Final-Only)", "#d1495b"),
}


def choose_best(df: pd.DataFrame):
    stats = df.groupby(["family", "reference_lr"], as_index=False).agg(
        mean_val_bpb=("val_bpb", "mean"),
        std_val_bpb=("val_bpb", lambda s: s.std(ddof=0)),
        n=("val_bpb", "size"),
    )
    stats["std_val_bpb"] = stats["std_val_bpb"].fillna(0.0)
    best = stats.sort_values(["family", "mean_val_bpb", "reference_lr"]).groupby("family", as_index=False).first()
    return stats, best


def plot_lr(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for family, (label, color) in LABELS.items():
        sub = stats[stats["family"] == family].sort_values("reference_lr")
        ax.errorbar(sub["reference_lr"], sub["mean_val_bpb"], yerr=sub["std_val_bpb"], marker="o", linewidth=2, capsize=4, color=color, label=label)
    ax.set_xlabel("reference_lr")
    ax.set_ylabel("val_bpb")
    ax.set_title("Main Matrix: val_bpb vs reference_lr")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_valbpb_by_lr.png")
    plt.close(fig)


def plot_curves(df: pd.DataFrame, best: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    summary = []
    for _, row in best.iterrows():
        family = row["family"]
        reference_lr = float(row["reference_lr"])
        label, color = LABELS[family]
        sub = df[(df["family"] == family) & (df["reference_lr"] == reference_lr)].sort_values("seed")
        merged = None
        for i, config_name in enumerate(sub["config"]):
            run = parse_log_curve(RUNS / f"{config_name}.log")[["step", "loss_ema"]].rename(columns={"loss_ema": f"run_{i}"})
            merged = run if merged is None else merged.merge(run, on="step", how="inner")
        values = merged.drop(columns=["step"])
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=0).fillna(0.0)
        ax.plot(merged["step"], mean, color=color, linewidth=2.2, label=f"{label} @ lr={reference_lr:.4g}")
        ax.fill_between(merged["step"], mean - std, mean + std, color=color, alpha=0.15)
        summary.append((family, reference_lr, float(mean.iloc[-1]), float(std.iloc[-1])))
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA train loss")
    ax.set_title("Main Matrix: Best Config per Family")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_loss_curves.png")
    plt.close(fig)
    return summary


def export_summaries(best: pd.DataFrame, curve_summary):
    best_lines = ["family\tbest_reference_lr\tmean_val_bpb\tstd_val_bpb\tn"]
    for _, row in best.iterrows():
        best_lines.append(f"{row['family']}\t{row['reference_lr']:.4g}\t{row['mean_val_bpb']:.6f}\t{row['std_val_bpb']:.6f}\t{int(row['n'])}")
    (ROOT / "results" / "main_results.tsv").write_text("\n".join(best_lines) + "\n")

    lines = [f"{family}\tlr={reference_lr:.4g}\tfinal_ema_loss_mean={end_mean:.6f}\tfinal_ema_loss_std={end_std:.6f}" for family, reference_lr, end_mean, end_std in curve_summary]
    (OUT / "fig_main_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    df["reference_lr"] = df["reference_lr"].astype(float)
    df["val_bpb"] = df["val_bpb"].astype(float)
    stats, best = choose_best(df)
    plot_lr(stats)
    curve_summary = plot_curves(df, best)
    export_summaries(best, curve_summary)


if __name__ == "__main__":
    main()
