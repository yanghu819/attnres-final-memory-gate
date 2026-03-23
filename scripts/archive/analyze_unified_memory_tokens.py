from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoresearch_attnres_project.metrics import mean_std

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/unified_memory_tokens_results.tsv'
OUT = ROOT / 'results/unified_memory_tokens_analysis.txt'
SUMMARY = ROOT / 'results/unified_memory_tokens_results.tsv'

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'unified_bigram_only_static',
    'unified_bigram_only_projected',
    'unified_ngram_static',
    'unified_ngram_projected',
]


def fmt(x: float) -> str:
    return f'{x:.6f}'


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    rows = []
    lines = []
    summary = {}
    for config in ORDER:
        sub = df[df['config'] == config]
        vals = pd.to_numeric(sub['val_bpb'], errors='coerce').dropna().tolist()
        if not vals:
            continue
        mean, std = mean_std(vals)
        mem_mean, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_total'], errors='coerce').dropna().tolist() or [0.0])
        proj_grad_mean, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_unified_proj_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        embed_grad_mean, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_embed_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        rows.append({
            'config': config,
            'mean_val_bpb': mean,
            'std_val_bpb': std,
            'mean_memory_total': mem_mean,
            'mean_embed_grad_norm': embed_grad_mean,
            'mean_unified_proj_grad_norm': proj_grad_mean,
        })
        summary[config] = {'val': mean, 'std': std, 'mem': mem_mean, 'egrad': embed_grad_mean, 'pgrad': proj_grad_mean}

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY, sep='\t', index=False)

    if 'strong_baseline' in summary:
        base = summary['strong_baseline']['val']
        for config in ORDER[1:]:
            if config in summary:
                delta = base - summary[config]['val']
                lines.append(f'{config}: mean={fmt(summary[config]["val"])} std={fmt(summary[config]["std"])} delta_vs_baseline={fmt(delta)} memory_total={fmt(summary[config]["mem"])} embed_grad={fmt(summary[config]["egrad"])} unified_proj_grad={fmt(summary[config]["pgrad"])}')

    if 'blend_ngram_ref' in summary and 'unified_ngram_static' in summary:
        delta = summary['blend_ngram_ref']['val'] - summary['unified_ngram_static']['val']
        lines.append(f'unified_ngram_static_vs_blend: {fmt(delta)} (negative means unified is worse)')
    if 'unified_ngram_static' in summary and 'unified_ngram_projected' in summary:
        delta = summary['unified_ngram_static']['val'] - summary['unified_ngram_projected']['val']
        lines.append(f'projected_minus_static_ngram: {fmt(delta)} (negative means projected is worse)')
    if 'unified_bigram_only_static' in summary and 'unified_ngram_static' in summary:
        delta = summary['unified_bigram_only_static']['val'] - summary['unified_ngram_static']['val']
        lines.append(f'unified_ngram_minus_unified_bigram_only_static: {fmt(delta)}')

    lines.append('')
    lines.append('Heuristics:')
    for config in ORDER[1:]:
        if config not in summary:
            continue
        mem = summary[config]['mem']
        pgrad = summary[config]['pgrad']
        egrad = summary[config]['egrad']
        if mem < 0.05:
            lines.append(f'- {config}: memory tokens are barely attended (mean mass {fmt(mem)}).')
        elif mem > 0.6:
            lines.append(f'- {config}: memory tokens absorb large final-mixer mass (mean mass {fmt(mem)}).')
        if pgrad == 0.0:
            lines.append(f'- {config}: projected bias path shows no gradient signal.')
        elif egrad > 0 and pgrad / egrad < 0.1:
            lines.append(f'- {config}: projected bias gradients are much smaller than memory-table gradients (ratio {fmt(pgrad / egrad)}).')

    OUT.write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
