from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/vres_memory_results.tsv'
FIGS = ROOT / 'results/figs'
RUNS = ROOT / 'runs/vres_memory'
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'modqkv_q002_kv005',
    'vres_tied_a001',
    'vres_tied_a002',
    'vres_tied_a005',
    'vres_tied_bigram_a002',
]
COLORS = {
    'strong_baseline': '#111111',
    'blend_ngram_ref': '#1f77b4',
    'modqkv_q002_kv005': '#9467bd',
    'vres_tied_a001': '#ff7f0e',
    'vres_tied_a002': '#2ca02c',
    'vres_tied_a005': '#d62728',
    'vres_tied_bigram_a002': '#8c564b',
}
LABELS = {
    'strong_baseline': 'AttnRes block2',
    'blend_ngram_ref': 'Blend ngram ref',
    'modqkv_q002_kv005': 'Organic ModQKV',
    'vres_tied_a001': 'V-res tied a=0.01',
    'vres_tied_a002': 'V-res tied a=0.02',
    'vres_tied_a005': 'V-res tied a=0.05',
    'vres_tied_bigram_a002': 'V-res tied+bigram a=0.02',
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
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=COLORS[config], alpha=0.15)
    ax.set_xlabel('step')
    ax.set_ylabel('train loss (EMA)')
    ax.set_title('Organic V-res AttnRes: loss curves')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_vres_memory_loss_curves.png', dpi=200)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ('val_bpb', 'val_bpb'),
        ('attnres_final_x0', 'final x0'),
        ('attnres_final_memory_vres_delta_norm', 'V-res delta norm'),
        ('attnres_final_memory_vres_proj_grad_norm', 'V-res proj grad'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        vals_by_cfg = []
        labels = []
        colors = []
        errs = []
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
    fig.savefig(FIGS / 'fig_vres_memory_summary.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_loss_curves(df)
    plot_summary(df)


if __name__ == '__main__':
    main()
