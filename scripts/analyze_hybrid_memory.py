from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoresearch_attnres_project.metrics import mean_std

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/hybrid_memory_results.tsv'
OUT = ROOT / 'results/hybrid_memory_analysis.txt'
SUMMARY = ROOT / 'results/hybrid_memory_results.tsv'

ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'hybrid_static_q002',
    'hybrid_static_q005',
    'hybrid_static_q010',
    'hybrid_projected_q005',
    'hybrid_projected_q010',
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
        fgate, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_gate'], errors='coerce').dropna().tolist() or [0.0])
        latest, _ = mean_std(pd.to_numeric(sub['attnres_final_latest'], errors='coerce').dropna().tolist() or [0.0])
        x0, _ = mean_std(pd.to_numeric(sub['attnres_final_x0'], errors='coerce').dropna().tolist() or [0.0])
        qproj, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_query_proj_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        egrad, _ = mean_std(pd.to_numeric(sub['attnres_final_memory_embed_grad_norm'], errors='coerce').dropna().tolist() or [0.0])
        rows.append({
            'config': config,
            'mean_val_bpb': mean,
            'std_val_bpb': std,
            'mean_final_memory_gate': fgate,
            'mean_query_gate': qgate,
            'mean_final_latest': latest,
            'mean_final_x0': x0,
            'mean_embed_grad_norm': egrad,
            'mean_query_proj_grad_norm': qproj,
        })
        summary[config] = dict(val=mean, std=std, fgate=fgate, qgate=qgate, latest=latest, x0=x0, egrad=egrad, qproj=qproj)
    pd.DataFrame(rows).to_csv(SUMMARY, sep='\t', index=False)

    if 'blend_ngram_ref' in summary:
        blend = summary['blend_ngram_ref']['val']
        lines.append(f'blend_ngram_ref: mean={fmt(summary["blend_ngram_ref"]["val"])} std={fmt(summary["blend_ngram_ref"]["std"])}')
        for config in ORDER:
            if config in {'strong_baseline', 'blend_ngram_ref'} or config not in summary:
                continue
            delta = blend - summary[config]['val']
            lines.append(
                f'{config}: mean={fmt(summary[config]["val"])} std={fmt(summary[config]["std"])} '
                f'delta_vs_blend={fmt(delta)} fgate={fmt(summary[config]["fgate"])} '
                f'qgate={fmt(summary[config]["qgate"])} final_latest={fmt(summary[config]["latest"])} '
                f'final_x0={fmt(summary[config]["x0"])} qproj_grad={fmt(summary[config]["qproj"])} embed_grad={fmt(summary[config]["egrad"])}'
            )
    lines.append('')
    lines.append('Heuristics:')
    for config in ORDER:
        if config not in summary or config in {'strong_baseline', 'blend_ngram_ref'}:
            continue
        qgate = summary[config]['qgate']
        x0 = summary[config]['x0']
        latest = summary[config]['latest']
        if qgate < 0.01:
            lines.append(f'- {config}: query hint is probably too weak (qgate {fmt(qgate)}).')
        elif qgate > 0.10:
            lines.append(f'- {config}: query hint is probably too strong (qgate {fmt(qgate)}).')
        if x0 > 0.8:
            lines.append(f'- {config}: final routing still collapses toward x0 (x0 {fmt(x0)}).')
        if latest < 0.05:
            lines.append(f'- {config}: final latest mass is very small (latest {fmt(latest)}).')
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
