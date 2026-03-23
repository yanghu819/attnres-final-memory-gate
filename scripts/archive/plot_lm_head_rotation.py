from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/lm_head_rotation_results.tsv'
RUNS = ROOT / 'runs/lm_head_rotation'
FIGS = ROOT / 'results/figs'
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    'strong_baseline_tied',
    'strong_baseline_untied',
    'lmrotate_r8_a005',
    'lmrotate_r16_a005',
    'lmrotate_r16_a010',
]
COLORS = {
    'strong_baseline_tied': '#111111',
    'strong_baseline_untied': '#666666',
    'lmrotate_r8_a005': '#1f77b4',
    'lmrotate_r16_a005': '#2ca02c',
    'lmrotate_r16_a010': '#d62728',
}
LABELS = {
    'strong_baseline_tied': 'Tied baseline',
    'strong_baseline_untied': 'Untied control',
    'lmrotate_r8_a005': 'Rotate r8 a0.05',
    'lmrotate_r16_a005': 'Rotate r16 a0.05',
    'lmrotate_r16_a010': 'Rotate r16 a0.10',
}


def plot_loss_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for config in ORDER:
        curves = []
        for row in df[df['config'] == config].itertuples(index=False):
            log_path = RUNS / f'{row.config}_seed{row.seed}.log'
            if log_path.exists():
                curve = parse_log_curve(log_path)
                if not curve.empty:
                    curves.append(curve[['step', 'loss_ema']].rename(columns={'loss_ema': f'seed{row.seed}'}).set_index('step'))
        if not curves:
            continue
        merged = pd.concat(curves, axis=1, join='outer').sort_index().interpolate(limit_direction='both')
        mean = merged.mean(axis=1)
        std = merged.std(axis=1, ddof=0).fillna(0.0)
        ax.plot(mean.index, mean.values, label=LABELS[config], color=COLORS[config], linewidth=2)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=COLORS[config], alpha=0.12)
    ax.set_xlabel('step')
    ax.set_ylabel('train loss (EMA)')
    ax.set_title('LM-head rotation on AttnRes strong baseline')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_lm_head_rotation_loss_curves.png', dpi=200)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ('val_bpb', 'val_bpb'),
        ('lm_head_rotation_lost_before', 'lost before'),
        ('lm_head_rotation_lost_after', 'lost after'),
        ('attnres_lm_head_grad_norm', 'lm_head grad norm'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        vals_by_cfg = []
        errs = []
        labels = []
        colors = []
        for config in ORDER:
            vals = pd.to_numeric(df.loc[df['config'] == config, col], errors='coerce').dropna().tolist()
            if not vals:
                continue
            mean, std = mean_std(vals)
            vals_by_cfg.append(mean)
            errs.append(std)
            labels.append(LABELS[config])
            colors.append(COLORS[config])
        ax.bar(range(len(vals_by_cfg)), vals_by_cfg, yerr=errs, color=colors, alpha=0.9)
        ax.set_xticks(range(len(vals_by_cfg)), labels, rotation=25, ha='right')
        ax.set_title(title)
        ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_lm_head_rotation_summary.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_loss_curves(df)
    plot_summary(df)


if __name__ == '__main__':
    main()
