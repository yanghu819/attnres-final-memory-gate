from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/hybrid_memory_results.tsv'
FIGS = ROOT / 'results/figs'
RUNS = ROOT / 'runs/hybrid_memory'
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'hybrid_static_q002',
    'hybrid_static_q005',
    'hybrid_static_q010',
    'hybrid_projected_q005',
    'hybrid_projected_q010',
]
COLORS = {
    'strong_baseline': '#111111',
    'blend_ngram_ref': '#1f77b4',
    'hybrid_static_q002': '#2ca02c',
    'hybrid_static_q005': '#17becf',
    'hybrid_static_q010': '#ff7f0e',
    'hybrid_projected_q005': '#d62728',
    'hybrid_projected_q010': '#9467bd',
}
LABELS = {
    'strong_baseline': 'AttnRes block2',
    'blend_ngram_ref': 'Blend ngram ref',
    'hybrid_static_q002': 'Hybrid static q=0.02',
    'hybrid_static_q005': 'Hybrid static q=0.05',
    'hybrid_static_q010': 'Hybrid static q=0.10',
    'hybrid_projected_q005': 'Hybrid proj q=0.05',
    'hybrid_projected_q010': 'Hybrid proj q=0.10',
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
    ax.set_title('Hybrid memory query + bounded blend')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_hybrid_memory_loss_curves.png', dpi=200)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ('val_bpb', 'val_bpb', False),
        ('attnres_final_memory_query_gate', 'query gate', False),
        ('attnres_final_memory_gate', 'blend gate', False),
        ('attnres_final_x0', 'final x0', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title, logy) in zip(axes.flat, metrics):
        xs, ys, es = [], [], []
        for i, config in enumerate(ORDER):
            vals = pd.to_numeric(df.loc[df['config'] == config, col], errors='coerce').dropna().tolist()
            if not vals:
                continue
            mean, std = mean_std(vals)
            xs.append(i)
            ys.append(mean)
            es.append(std)
        colors = [COLORS[ORDER[i]] for i in xs]
        labels = [LABELS[ORDER[i]] for i in xs]
        ax.bar(range(len(xs)), ys, yerr=es, color=colors, alpha=0.9)
        ax.set_xticks(range(len(xs)), labels, rotation=25, ha='right')
        ax.set_title(title)
        if logy:
            ax.set_yscale('log')
        ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_hybrid_memory_summary.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_loss_curves(df)
    plot_summary(df)


if __name__ == '__main__':
    main()
