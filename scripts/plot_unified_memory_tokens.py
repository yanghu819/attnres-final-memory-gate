from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/unified_memory_tokens_results.tsv'
FIGS = ROOT / 'results/figs'
RUNS = ROOT / 'runs/unified_memory_tokens'
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'unified_bigram_only_static',
    'unified_bigram_only_projected',
    'unified_ngram_static',
    'unified_ngram_projected',
]
COLORS = {
    'strong_baseline': '#111111',
    'blend_ngram_ref': '#1f77b4',
    'unified_bigram_only_static': '#ff7f0e',
    'unified_bigram_only_projected': '#d62728',
    'unified_ngram_static': '#2ca02c',
    'unified_ngram_projected': '#9467bd',
}
LABELS = {
    'strong_baseline': 'AttnRes block2',
    'blend_ngram_ref': 'Blend unigram+bigram',
    'unified_bigram_only_static': 'Unified bigram static',
    'unified_bigram_only_projected': 'Unified bigram projected',
    'unified_ngram_static': 'Unified unigram+bigram static',
    'unified_ngram_projected': 'Unified unigram+bigram projected',
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
    ax.set_title('Unified Memory Tokens on AttnRes: loss curves')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_unified_memory_tokens_loss_curves.png', dpi=200)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ('val_bpb', 'val_bpb', False),
        ('attnres_final_memory_total', 'memory mass', False),
        ('attnres_final_memory_unified_proj_grad_norm', 'proj grad norm', True),
        ('attnres_final_memory_embed_grad_norm', 'memory grad norm', True),
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
    fig.savefig(FIGS / 'fig_unified_memory_tokens_summary.png', dpi=200)
    plt.close(fig)


def plot_weights(df: pd.DataFrame) -> None:
    cfgs = [c for c in ORDER if c != 'strong_baseline']
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0.0] * len(cfgs)
    for col, title, color in [
        ('attnres_final_memory_unigram', 'unigram', '#1f77b4'),
        ('attnres_final_memory_bigram_bank0', 'bank0', '#ff7f0e'),
        ('attnres_final_memory_bigram_bank1', 'bank1', '#2ca02c'),
        ('attnres_final_memory_bigram_bank2', 'bank2', '#d62728'),
        ('attnres_final_memory_bigram_bank3', 'bank3', '#9467bd'),
    ]:
        vals = []
        for config in cfgs:
            s = pd.to_numeric(df.loc[df['config'] == config, col], errors='coerce').dropna()
            vals.append(float(s.mean()) if len(s) else 0.0)
        ax.bar(range(len(cfgs)), vals, bottom=bottoms, label=title, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(cfgs)), [LABELS[c] for c in cfgs], rotation=25, ha='right')
    ax.set_ylabel('mean attention mass')
    ax.set_title('Unified memory source weights')
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_unified_memory_tokens_weights.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_loss_curves(df)
    plot_summary(df)
    plot_weights(df)

if __name__ == '__main__':
    main()
