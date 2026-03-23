from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoresearch_attnres_project.metrics import mean_std

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/model_size_holdout_results.tsv'
SUMMARY = ROOT / 'results/model_size_holdout_results.tsv'
OUT = ROOT / 'results/model_size_holdout_analysis.txt'

SIZE_ORDER = ['s384_d6', 'm512_d8', 'l640_d10']
FAMILY_ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'unified_ngram_projected',
    'query_bigram_only_projected',
    'query_ngram_projected',
]


def fmt(x: float) -> str:
    return f'{x:.6f}'


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    rows = []
    lines = []
    for size in SIZE_ORDER:
        size_df = df[df['size_tag'] == size]
        if size_df.empty:
            continue
        lines.append(f'[{size}]')
        vals_by_cfg = {}
        for cfg in FAMILY_ORDER:
            sub = size_df[size_df['config'] == cfg]
            vals = pd.to_numeric(sub['val_bpb'], errors='coerce').dropna().tolist()
            if not vals:
                continue
            mean, std = mean_std(vals)
            vals_by_cfg[cfg] = mean
            rows.append({
                'size_tag': size,
                'config': cfg,
                'mean_val_bpb': mean,
                'std_val_bpb': std,
                'mean_peak_vram_mb': float(pd.to_numeric(sub['peak_vram_mb'], errors='coerce').mean()),
                'mean_final_x0': float(pd.to_numeric(sub['attnres_final_x0'], errors='coerce').mean()),
                'mean_final_latest': float(pd.to_numeric(sub['attnres_final_latest'], errors='coerce').mean()),
                'mean_final_memory_total': float(pd.to_numeric(sub['attnres_final_memory_total'], errors='coerce').mean()),
                'mean_query_gate': float(pd.to_numeric(sub['attnres_final_memory_query_gate'], errors='coerce').mean()),
            })
            lines.append(f'  {cfg}: mean={fmt(mean)} std={fmt(std)}')
        if 'strong_baseline' in vals_by_cfg and 'blend_ngram_ref' in vals_by_cfg:
            lines.append(f'  blend_vs_baseline={fmt(vals_by_cfg["strong_baseline"] - vals_by_cfg["blend_ngram_ref"])}')
        if 'blend_ngram_ref' in vals_by_cfg and 'unified_ngram_projected' in vals_by_cfg:
            lines.append(f'  unified_vs_blend={fmt(vals_by_cfg["blend_ngram_ref"] - vals_by_cfg["unified_ngram_projected"])} (negative means unified worse)')
        if 'blend_ngram_ref' in vals_by_cfg and 'query_bigram_only_projected' in vals_by_cfg:
            lines.append(f'  query_bigram_vs_blend={fmt(vals_by_cfg["blend_ngram_ref"] - vals_by_cfg["query_bigram_only_projected"])} (negative means query worse)')
        if 'blend_ngram_ref' in vals_by_cfg and 'query_ngram_projected' in vals_by_cfg:
            lines.append(f'  query_ngram_vs_blend={fmt(vals_by_cfg["blend_ngram_ref"] - vals_by_cfg["query_ngram_projected"])} (negative means query worse)')
        qn = size_df[size_df['config'] == 'query_ngram_projected']
        if not qn.empty:
            x0 = float(pd.to_numeric(qn['attnres_final_x0'], errors='coerce').mean())
            latest = float(pd.to_numeric(qn['attnres_final_latest'], errors='coerce').mean())
            entropy = float(pd.to_numeric(qn['attnres_final_entropy'], errors='coerce').mean())
            lines.append(f'  query_ngram_projected_routing: x0={fmt(x0)} latest={fmt(latest)} entropy={fmt(entropy)}')
        uni = size_df[size_df['config'] == 'unified_ngram_projected']
        if not uni.empty:
            mem = float(pd.to_numeric(uni['attnres_final_memory_total'], errors='coerce').mean())
            x0 = float(pd.to_numeric(uni['attnres_final_x0'], errors='coerce').mean())
            latest = float(pd.to_numeric(uni['attnres_final_latest'], errors='coerce').mean())
            lines.append(f'  unified_ngram_projected_memory: total={fmt(mem)} x0={fmt(x0)} latest={fmt(latest)}')
        lines.append('')

    pd.DataFrame(rows).to_csv(SUMMARY, sep='\t', index=False)
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
