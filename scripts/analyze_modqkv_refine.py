from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/modqkv_refine_results.tsv'
OUT = ROOT / 'results/modqkv_refine_analysis.txt'
TSV = ROOT / 'results/modqkv_refine_results.tsv'

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


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    rows = []
    for config in ORDER:
        sub = df[df['config'] == config].copy()
        if sub.empty:
            continue
        val = pd.to_numeric(sub['val_bpb'], errors='coerce')
        rows.append({
            'config': config,
            'count': int(val.notna().sum()),
            'mean_val_bpb': float(val.mean()),
            'std_val_bpb': float(val.std(ddof=0)),
            'mean_final_latest': float(pd.to_numeric(sub['attnres_final_latest'], errors='coerce').mean()),
            'mean_final_x0': float(pd.to_numeric(sub['attnres_final_x0'], errors='coerce').mean()),
            'mean_final_entropy': float(pd.to_numeric(sub['attnres_final_entropy'], errors='coerce').mean()),
            'mean_q_delta_norm': float(pd.to_numeric(sub['attnres_final_memory_mod_q_delta_norm'], errors='coerce').mean()),
            'mean_k_delta_norm': float(pd.to_numeric(sub['attnres_final_memory_mod_k_delta_norm'], errors='coerce').mean()),
            'mean_v_delta_norm': float(pd.to_numeric(sub['attnres_final_memory_mod_v_delta_norm'], errors='coerce').mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TSV, sep='\t', index=False)
    modqkv_ref = float(summary.loc[summary['config'] == 'modqkv_q002_k005_v005', 'mean_val_bpb'].iloc[0]) if (summary['config'] == 'modqkv_q002_k005_v005').any() else float('nan')
    best_row = summary.loc[summary['mean_val_bpb'].idxmin()]
    lines = [
        f"best_config\t{best_row['config']}",
        f"best_mean_val_bpb\t{best_row['mean_val_bpb']:.6f}",
    ]
    if pd.notna(modqkv_ref):
        lines.append(f"delta_vs_modqkv_ref\t{best_row['mean_val_bpb'] - modqkv_ref:.6f}")
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.config}\tmean={row.mean_val_bpb:.6f}\tstd={row.std_val_bpb:.6f}\tlatest={row.mean_final_latest:.6f}\tx0={row.mean_final_x0:.6f}\tentropy={row.mean_final_entropy:.6f}\tq={row.mean_q_delta_norm:.6f}\tk={row.mean_k_delta_norm:.6f}\tv={row.mean_v_delta_norm:.6f}"
        )
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
