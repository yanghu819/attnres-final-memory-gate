#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results' / 'raw' / 'ngram_module_ablation_results.tsv'
OUT = ROOT / 'results' / 'ngram_module_ablation_analysis.txt'

LABELS = {
    'strong_baseline': 'AttnRes Block2',
    'unigram_only_matched': 'Unigram only (cap 0.85)',
    'unigram_only_best': 'Unigram only (best unigram)',
    'bigram_only_4x1m': 'Bigram only (4x1M)',
    'unigram_plus_bigram_4x1m': 'Unigram + Bigram (4x1M)',
}


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    df['val_bpb'] = df['val_bpb'].astype(float)
    summary = (
        df.groupby('config', sort=False)['val_bpb']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values('mean')
    )
    best = summary.iloc[0]
    baseline = summary[summary['config'] == 'strong_baseline'].iloc[0]
    lines = []
    lines.append('N-gram Module Ablation Summary')
    lines.append('')
    lines.append(f"best_config\t{best['config']}\t{LABELS.get(best['config'], best['config'])}")
    lines.append(f"best_mean_val_bpb\t{best['mean']:.6f}")
    lines.append(f"baseline_mean_val_bpb\t{baseline['mean']:.6f}")
    lines.append(f"best_minus_baseline\t{best['mean'] - baseline['mean']:.6f}")
    lines.append('')
    lines.append('Per-family summary')
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.config}\t{LABELS.get(row.config, row.config)}\tmean={row.mean:.6f}\tstd={0.0 if pd.isna(row.std) else row.std:.6f}\tn={int(row.count)}"
        )
    if {'bigram_only_4x1m', 'unigram_plus_bigram_4x1m'}.issubset(set(summary['config'])):
        b = float(summary[summary['config'] == 'bigram_only_4x1m']['mean'].iloc[0])
        ub = float(summary[summary['config'] == 'unigram_plus_bigram_4x1m']['mean'].iloc[0])
        lines.append('')
        lines.append(f"unigram_plus_minus_bigram_only\t{ub - b:.6f}")
    OUT.write_text('\n'.join(lines) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
