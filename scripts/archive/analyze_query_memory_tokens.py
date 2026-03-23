from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoresearch_attnres_project.metrics import mean_std

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/query_memory_tokens_results.tsv'
OUT = ROOT / 'results/query_memory_tokens_analysis.txt'
SUMMARY = ROOT / 'results/query_memory_tokens_results.tsv'

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'query_bigram_only_static',
    'query_bigram_only_projected',
    'query_ngram_static',
    'query_ngram_projected',
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
        qgate, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_query_gate'], errors='coerce').dropna().tolist() or [0.0])
        qnorm, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_query_norm'], errors='coerce').dropna().tolist() or [0.0])
        latest, _ = mean_std(pd.to_numeric(sub['attnres_final_latest'], errors='coerce').dropna().tolist() or [0.0])
        x0, _ = mean_std(pd.to_numeric(sub['attnres_final_x0'], errors='coerce').dropna().tolist() or [0.0])
        proj_grad, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_gate_proj_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        embed_grad, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_embed_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        rows.append({
            'config': config,
            'mean_val_bpb': mean,
            'std_val_bpb': std,
            'mean_query_gate': qgate,
            'mean_query_norm': qnorm,
            'mean_final_latest': latest,
            'mean_final_x0': x0,
            'mean_embed_grad_norm': embed_grad,
            'mean_gate_proj_grad_norm': proj_grad,
        })
        summary[config] = {
            'val': mean, 'std': std, 'qgate': qgate, 'qnorm': qnorm,
            'latest': latest, 'x0': x0, 'egrad': embed_grad, 'pgrad': proj_grad,
        }

    pd.DataFrame(rows).to_csv(SUMMARY, sep='\t', index=False)

    if 'strong_baseline' in summary:
        base = summary['strong_baseline']['val']
        for config in ORDER[1:]:
            if config in summary:
                delta = base - summary[config]['val']
                lines.append(
                    f'{config}: mean={fmt(summary[config]["val"])} std={fmt(summary[config]["std"])} '
                    f'delta_vs_baseline={fmt(delta)} qgate={fmt(summary[config]["qgate"])} '
                    f'qnorm={fmt(summary[config]["qnorm"])} final_latest={fmt(summary[config]["latest"])} '
                    f'final_x0={fmt(summary[config]["x0"])} embed_grad={fmt(summary[config]["egrad"])} '
                    f'gate_proj_grad={fmt(summary[config]["pgrad"])}'
                )

    if 'blend_ngram_ref' in summary and 'query_ngram_static' in summary:
        delta = summary['blend_ngram_ref']['val'] - summary['query_ngram_static']['val']
        lines.append(f'query_ngram_static_vs_blend: {fmt(delta)} (negative means query is worse)')
    if 'query_ngram_static' in summary and 'query_ngram_projected' in summary:
        delta = summary['query_ngram_static']['val'] - summary['query_ngram_projected']['val']
        lines.append(f'query_projected_minus_static_ngram: {fmt(delta)} (negative means projected is worse)')
    if 'query_bigram_only_static' in summary and 'query_ngram_static' in summary:
        delta = summary['query_bigram_only_static']['val'] - summary['query_ngram_static']['val']
        lines.append(f'query_ngram_minus_query_bigram_only_static: {fmt(delta)}')

    lines.append('')
    lines.append('Heuristics:')
    for config in ORDER[1:]:
        if config not in summary:
            continue
        qgate = summary[config]['qgate']
        qnorm = summary[config]['qnorm']
        pgrad = summary[config]['pgrad']
        egrad = summary[config]['egrad']
        if qgate < 0.05 or qnorm < 0.1:
            lines.append(f'- {config}: memory query contribution is weak (gate {fmt(qgate)}, norm {fmt(qnorm)}).')
        elif qgate > 0.6:
            lines.append(f'- {config}: memory query strongly steers final mixer (gate {fmt(qgate)}).')
        if pgrad == 0.0:
            lines.append(f'- {config}: projected query path shows no gradient signal.')
        elif egrad > 0 and pgrad / egrad < 0.1:
            lines.append(f'- {config}: projected query gradients are much smaller than memory-table gradients (ratio {fmt(pgrad / egrad)}).')

    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
