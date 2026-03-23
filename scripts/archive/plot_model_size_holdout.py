from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/model_size_holdout_results.tsv'
RUNS = ROOT / 'runs/model_size_holdout'
FIGS = ROOT / 'results/figs'
FIGS.mkdir(parents=True, exist_ok=True)

SIZE_ORDER = ['s384_d6', 'm512_d8', 'l640_d10']
SIZE_LABELS = {'s384_d6': 'd6 / 384d', 'm512_d8': 'd8 / 512d', 'l640_d10': 'd10 / 640d'}
FAMILY_ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'unified_ngram_projected',
    'query_bigram_only_projected',
    'query_ngram_projected',
]
COLORS = {
    'strong_baseline': '#111111',
    'blend_ngram_ref': '#1f77b4',
    'unified_ngram_projected': '#d62728',
    'query_bigram_only_projected': '#2ca02c',
    'query_ngram_projected': '#9467bd',
}
LABELS = {
    'strong_baseline': 'AttnRes block2',
    'blend_ngram_ref': 'Blend unigram+bigram',
    'unified_ngram_projected': 'Unified ngram projected',
    'query_bigram_only_projected': 'Query bigram projected',
    'query_ngram_projected': 'Query ngram projected',
}


def plot_valbpb(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, size in zip(axes, SIZE_ORDER):
        sub = df[df['size_tag'] == size]
        xs, ys, es, colors, labels = [], [], [], [], []
        for i, cfg in enumerate(FAMILY_ORDER):
            vals = pd.to_numeric(sub.loc[sub['config'] == cfg, 'val_bpb'], errors='coerce').dropna().tolist()
            if not vals:
                continue
            mean, std = mean_std(vals)
            xs.append(len(xs))
            ys.append(mean)
            es.append(std)
            colors.append(COLORS[cfg])
            labels.append(LABELS[cfg])
        ax.bar(range(len(xs)), ys, yerr=es, color=colors, alpha=0.9)
        ax.set_xticks(range(len(xs)), labels, rotation=25, ha='right')
        ax.set_title(SIZE_LABELS[size])
        ax.grid(alpha=0.2, axis='y')
    axes[0].set_ylabel('val_bpb')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_model_size_holdout_valbpb.png', dpi=200)
    plt.close(fig)


def plot_mechanism(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(SIZE_ORDER))
    for cfg in ['unified_ngram_projected', 'query_ngram_projected']:
        mem = []
        x0 = []
        for size in SIZE_ORDER:
            sub = df[(df['size_tag'] == size) & (df['config'] == cfg)]
            mem.append(float(pd.to_numeric(sub['attnres_final_memory_total'], errors='coerce').mean()) if not sub.empty else float('nan'))
            x0.append(float(pd.to_numeric(sub['attnres_final_x0'], errors='coerce').mean()) if not sub.empty else float('nan'))
        axes[0].plot(list(x), mem, marker='o', label=LABELS[cfg], color=COLORS[cfg])
        axes[1].plot(list(x), x0, marker='o', label=LABELS[cfg], color=COLORS[cfg])
    for ax, title, ylabel in [
        (axes[0], 'Memory mass vs size', 'mean memory mass'),
        (axes[1], 'Final x0 weight vs size', 'mean final x0 weight'),
    ]:
        ax.set_xticks(list(x), [SIZE_LABELS[s] for s in SIZE_ORDER], rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_model_size_holdout_mechanism.png', dpi=200)
    plt.close(fig)


def plot_loss_curves(df: pd.DataFrame) -> None:
    # Use largest size only to keep figure readable.
    target_size = 'l640_d10'
    sub = df[df['size_tag'] == target_size]
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg in FAMILY_ORDER:
        curves = []
        for row in sub[sub['config'] == cfg].itertuples(index=False):
            log_path = RUNS / f'{row.size_tag}_{row.config}_seed{row.seed}.log'
            if log_path.exists():
                curve = parse_log_curve(log_path)
                if not curve.empty:
                    curves.append(curve[['step', 'loss_ema']].rename(columns={'loss_ema': f'seed{row.seed}'}).set_index('step'))
        if not curves:
            continue
        merged = pd.concat(curves, axis=1, join='outer').sort_index().interpolate(limit_direction='both')
        mean = merged.mean(axis=1)
        std = merged.std(axis=1, ddof=0).fillna(0.0)
        ax.plot(mean.index, mean.values, label=LABELS[cfg], color=COLORS[cfg], linewidth=2)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, color=COLORS[cfg], alpha=0.15)
    ax.set_xlabel('step')
    ax.set_ylabel('train loss (EMA)')
    ax.set_title('Largest-size holdout curves (d10 / 640d)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_model_size_holdout_loss_curves.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_valbpb(df)
    plot_mechanism(df)
    plot_loss_curves(df)


if __name__ == '__main__':
    main()
