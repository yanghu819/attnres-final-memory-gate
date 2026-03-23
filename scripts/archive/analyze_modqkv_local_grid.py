from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/modqkv_local_grid_results.tsv'
OUT = ROOT / 'results/modqkv_local_grid_analysis.txt'
TSV = ROOT / 'results/modqkv_local_grid_results.tsv'


def decode_tag(tag: str) -> float:
    m = re.match(r'^0*(\d+)$', tag)
    if m is None:
        raise ValueError(f'bad scale tag: {tag}')
    digits = m.group(1)
    if len(digits) <= 2:
        return float(digits) / 100.0
    if len(digits) == 3:
        return float(digits) / 100.0
    return float(digits) / 1000.0


def config_sort_key(name: str) -> tuple[int, float, float, float]:
    if name == 'strong_baseline':
        return (0, 0.0, 0.0, 0.0)
    if name == 'blend_ngram_ref':
        return (1, 0.0, 0.0, 0.0)
    parts = name.split('_')
    q = decode_tag(parts[1][1:])
    k = decode_tag(parts[2][1:])
    v = decode_tag(parts[3][1:])
    return (2, q, k, v)


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    rows = []
    for config in sorted(df['config'].unique(), key=config_sort_key):
        sub = df[df['config'] == config].copy()
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
    modqkv_ref = float(summary.loc[summary['config'] == 'modqkv_q0020_k005_v005', 'mean_val_bpb'].iloc[0]) if (summary['config'] == 'modqkv_q0020_k005_v005').any() else float('nan')
    best_mod = summary[~summary['config'].isin(['strong_baseline', 'blend_ngram_ref'])].sort_values('mean_val_bpb').iloc[0]
    lines = [
        f"best_modqkv_config\t{best_mod['config']}",
        f"best_modqkv_mean_val_bpb\t{best_mod['mean_val_bpb']:.6f}",
    ]
    if pd.notna(modqkv_ref):
        lines.append(f"delta_vs_modqkv_q0020_k005_v005\t{best_mod['mean_val_bpb'] - modqkv_ref:.6f}")
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.config}\tmean={row.mean_val_bpb:.6f}\tstd={row.std_val_bpb:.6f}\tlatest={row.mean_final_latest:.6f}\tx0={row.mean_final_x0:.6f}\tentropy={row.mean_final_entropy:.6f}\tq={row.mean_q_delta_norm:.6f}\tk={row.mean_k_delta_norm:.6f}\tv={row.mean_v_delta_norm:.6f}"
        )
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
