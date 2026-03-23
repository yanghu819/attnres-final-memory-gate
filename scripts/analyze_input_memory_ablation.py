#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results' / 'raw' / 'input_memory_ablation_results.tsv'
OUT = ROOT / 'results' / 'input_memory_ablation_analysis.txt'

LABELS = {
    'strong_baseline': 'AttnRes Block2',
    'blend_ngram_ref': 'Final Blend (Uni+Bi)',
    'input_bigram_hash': 'Input BigramHash',
    'input_bigram_hash_smear': 'Input BigramHash + SmearGate',
    'input_bigram_trigram_hash': 'Input Bigram+Trigram',
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
    lines = [
        'Input Memory Ablation Summary',
        '',
        f"best_config\t{best['config']}\t{LABELS.get(best['config'], best['config'])}",
        f"best_mean_val_bpb\t{best['mean']:.6f}",
        f"baseline_mean_val_bpb\t{baseline['mean']:.6f}",
        f"best_minus_baseline\t{best['mean'] - baseline['mean']:.6f}",
        '',
        'Per-family summary',
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.config}\t{LABELS.get(row.config, row.config)}\tmean={row.mean:.6f}\tstd={0.0 if pd.isna(row.std) else row.std:.6f}\tn={int(row.count)}"
        )
    OUT.write_text('\n'.join(lines) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
