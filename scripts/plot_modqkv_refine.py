from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from autoresearch_attnres_project.metrics import mean_std, parse_log_curve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/modqkv_refine_results.tsv'
FIGS = ROOT / 'results/figs'
RUNS = ROOT / 'runs/modqkv_refine'
FIGS.mkdir(parents=True, exist_ok=True)

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'modqkv_q001_k002_v002',
    'modqkv_q002_k005_v005',
    'modqkv_q002_k005_v002',
    'modqkv_q002_k008_v002',
    'modqkv_q003_k008_v002',
    'modqkv_q003_k010_v002',
    'modqkv_q004_k010_v002',
]
COLORS = {
    'strong_baseline': '#111111',
    'blend_ngram_ref': '#1f77b4',
    'modqkv_q001_k002_v002': '#ff7f0e',
    'modqkv_q002_k005_v005': '#2ca02c',
    'modqkv_q002_k005_v002': '#d62728',
    'modqkv_q002_k008_v002': '#9467bd',
    'modqkv_q003_k008_v002': '#8c564b',
    'modqkv_q003_k010_v002': '#e377c2',
    'modqkv_q004_k010_v002': '#17becf',
}
LABELS = {
    'strong_baseline': 'AttnRes block2',
    'blend_ngram_ref': 'Blend ref',
    'modqkv_q001_k002_v002': 'q0.01 k0.02 v0.02',
    'modqkv_q002_k005_v005': 'q0.02 k0.05 v0.05',
    'modqkv_q002_k005_v002': 'q0.02 k0.05 v0.02',
    'modqkv_q002_k008_v002': 'q0.02 k0.08 v0.02',
    'modqkv_q003_k008_v002': 'q0.03 k0.08 v0.02',
    'modqkv_q003_k010_v002': 'q0.03 k0.10 v0.02',
    'modqkv_q004_k010_v002': 'q0.04 k0.10 v0.02',
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
    ax.set_title('Organic ModQKV refine: loss curves')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_modqkv_refine_loss_curves.png', dpi=200)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ('val_bpb', 'val_bpb'),
        ('attnres_final_x0', 'final x0'),
        ('attnres_final_memory_mod_q_delta_norm', 'q delta norm'),
        ('attnres_final_memory_mod_v_delta_norm', 'v delta norm'),
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
    fig.savefig(FIGS / 'fig_modqkv_refine_summary.png', dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    plot_loss_curves(df)
    plot_summary(df)


if __name__ == '__main__':
    main()
