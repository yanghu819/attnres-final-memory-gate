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

RESULTS = ROOT / "results" / "raw" / "followup_results.tsv"
RUNS = ROOT / "runs" / "followup"
OUT = ROOT / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "attnres_block2": ("AttnRes Block2 Strong Baseline", "#1f3b4d"),
    "projected_deepemb_all": ("Projected Token Memory (All)", "#00798c"),
    "projected_deepemb_final": ("Projected Token Memory (Final-Only)", "#edae49"),
}


def load_group(df: pd.DataFrame, family: str):
    sub = df[df["family"] == family].copy().sort_values("seed")
    sub["val_bpb"] = sub["val_bpb"].astype(float)
    return sub, [RUNS / f"{name}.log" for name in sub["config"]]


def plot_curves(loaded):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    summary = []
    for family, (label, color) in GROUPS.items():
        _, logs = loaded[family]
        merged = None
        for i, log_path in enumerate(logs):
            run = parse_log_curve(log_path)[["step", "loss_ema"]].rename(columns={"loss_ema": f"run_{i}"})
            merged = run if merged is None else merged.merge(run, on="step", how="inner")
        values = merged.drop(columns=["step"])
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=0).fillna(0.0)
        ax.plot(merged["step"], mean, color=color, linewidth=2.2, label=f"{label} (n={len(logs)})")
        ax.fill_between(merged["step"], mean - std, mean + std, color=color, alpha=0.15)
        summary.append((family, float(mean.iloc[-1]), float(std.iloc[-1])))
    ax.set_title("Follow-Up: Fresh Seeds 45-48")
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA train loss")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_followup_loss_curves.png")
    plt.close(fig)
    return summary


def plot_delta(loaded):
    control_logs = loaded["attnres_block2"][1]
    merged_control = None
    for i, log_path in enumerate(control_logs):
        run = parse_log_curve(log_path)[["step", "loss_ema"]].rename(columns={"loss_ema": f"run_{i}"})
        merged_control = run if merged_control is None else merged_control.merge(run, on="step", how="inner")
    control_mean = merged_control.drop(columns=["step"]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    for family, (label, color) in GROUPS.items():
        if family == "attnres_block2":
            continue
        _, logs = loaded[family]
        merged = None
        for i, log_path in enumerate(logs):
            run = parse_log_curve(log_path)[["step", "loss_ema"]].rename(columns={"loss_ema": f"run_{i}"})
            merged = run if merged is None else merged.merge(run, on="step", how="inner")
        common = merged.merge(pd.DataFrame({"step": merged_control["step"], "control_mean": control_mean}), on="step", how="inner")
        mean = common.drop(columns=["step", "control_mean"]).mean(axis=1)
        delta = mean - common["control_mean"]
        ax.plot(common["step"], delta, color=color, linewidth=2.2, label=label)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_title("Follow-Up: EMA Loss Delta vs Strong Baseline")
    ax.set_xlabel("Step")
    ax.set_ylabel("EMA loss delta")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_followup_delta_vs_control.png")
    plt.close(fig)


def plot_val(loaded):
    order = list(GROUPS)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
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
        offsets = [-0.09, -0.03, 0.03, 0.09][: len(df)]
        for xoff, y in zip(offsets, df["val_bpb"].tolist()):
            ax.scatter(i + xoff, y, color="#111111", s=18, alpha=0.75, zorder=3)
    ax.set_xticks(list(xs), [GROUPS[k][0] for k in order], rotation=0)
    ax.set_ylabel("val_bpb")
    ax.set_title("Follow-Up: val_bpb on Seeds 45-48")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_followup_valbpb.png")
    plt.close(fig)


def export_summary(loaded, curve_summary):
    lines = ["family\tmean_val_bpb\tstd_val_bpb\tn"]
    for family in GROUPS:
        df, _ = loaded[family]
        values = df["val_bpb"].tolist()
        mean = sum(values) / len(values)
        std = pd.Series(values).std(ddof=0)
        lines.append(f"{family}\t{mean:.6f}\t{std:.6f}\t{len(values)}")
    (ROOT / "results" / "followup_results.tsv").write_text("\n".join(lines) + "\n")

    curve_lines = [f"{family}\tfinal_ema_loss_mean={mean:.6f}\tfinal_ema_loss_std={std:.6f}" for family, mean, std in curve_summary]
    (OUT / "fig_followup_summary.txt").write_text("\n".join(curve_lines) + "\n")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    loaded = {family: load_group(df, family) for family in GROUPS}
    curve_summary = plot_curves(loaded)
    plot_delta(loaded)
    plot_val(loaded)
    export_summary(loaded, curve_summary)


if __name__ == "__main__":
    main()
