from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/modqkv_local_grid_results.tsv'
FIGS = ROOT / 'results/figs'
FIGS.mkdir(parents=True, exist_ok=True)


def heatmap(df: pd.DataFrame, v_target: float) -> pd.DataFrame:
    rows = []
    for config, sub in df.groupby('config'):
        if config in {'strong_baseline', 'blend_ngram_ref'}:
            continue
        q = float(pd.to_numeric(sub['q_mod_scale'], errors='coerce').iloc[0])
        k = float(pd.to_numeric(sub['k_mod_scale'], errors='coerce').iloc[0])
        v = float(pd.to_numeric(sub['v_mod_scale'], errors='coerce').iloc[0])
        if abs(v - v_target) > 1e-9:
            continue
        rows.append({
            'q': q,
            'k': k,
            'mean_val_bpb': pd.to_numeric(sub['val_bpb'], errors='coerce').mean(),
        })
    return pd.DataFrame(rows).pivot(index='q', columns='k', values='mean_val_bpb').sort_index().sort_index(axis=1)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, v_target in zip(axes, [0.04, 0.05, 0.06]):
        pivot = heatmap(df, v_target)
        im = ax.imshow(pivot.values, cmap='viridis_r')
        ax.set_xticks(range(len(pivot.columns)), [f'{k:.2f}' for k in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f'{q:.3f}' for q in pivot.index])
        ax.set_xlabel('k scale')
        ax.set_ylabel('q scale')
        ax.set_title(f'val_bpb @ v={v_target:.2f}')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label='mean val_bpb')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_modqkv_local_grid_heatmaps.png', dpi=200)
    plt.close(fig)

    summary = []
    for config in sorted(df['config'].unique()):
        vals = pd.to_numeric(df.loc[df['config'] == config, 'val_bpb'], errors='coerce').dropna()
        summary.append((config, vals.mean()))
    summary = sorted(summary, key=lambda x: x[1])
    top = summary[:12]
    labels = [cfg for cfg, _ in top]
    values = [val for _, val in top]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(values)), values, color=['#111111' if x == 'strong_baseline' else '#1f77b4' if x == 'blend_ngram_ref' else '#2ca02c' for x in labels])
    ax.set_xticks(range(len(values)), labels, rotation=30, ha='right')
    ax.set_ylabel('mean val_bpb')
    ax.set_title('Top modqkv local-grid configs')
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_modqkv_local_grid_topbars.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
