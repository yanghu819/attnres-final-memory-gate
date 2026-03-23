from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/lm_head_rotation_results.tsv'
OUT = ROOT / 'results/lm_head_rotation_analysis.txt'
TSV = ROOT / 'results/lm_head_rotation_results.tsv'

ORDER = [
    'strong_baseline_tied',
    'strong_baseline_untied',
    'lmrotate_r8_a005',
    'lmrotate_r16_a005',
    'lmrotate_r16_a010',
]


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    rows = []
    for config in ORDER:
        sub = df[df['config'] == config].copy()
        if sub.empty:
            continue
        rows.append({
            'config': config,
            'count': int(pd.to_numeric(sub['val_bpb'], errors='coerce').notna().sum()),
            'mean_val_bpb': float(pd.to_numeric(sub['val_bpb'], errors='coerce').mean()),
            'std_val_bpb': float(pd.to_numeric(sub['val_bpb'], errors='coerce').std(ddof=0)),
            'mean_lost_before': float(pd.to_numeric(sub['lm_head_rotation_lost_before'], errors='coerce').mean()),
            'mean_lost_after': float(pd.to_numeric(sub['lm_head_rotation_lost_after'], errors='coerce').mean()),
            'mean_rotation_updates': float(pd.to_numeric(sub['lm_head_rotation_updates'], errors='coerce').mean()),
            'mean_lm_head_grad_norm': float(pd.to_numeric(sub['attnres_lm_head_grad_norm'], errors='coerce').mean()),
            'mean_wte_grad_norm': float(pd.to_numeric(sub['attnres_wte_grad_norm'], errors='coerce').mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TSV, sep='\t', index=False)

    tied = summary.loc[summary['config'] == 'strong_baseline_tied', 'mean_val_bpb']
    untied = summary.loc[summary['config'] == 'strong_baseline_untied', 'mean_val_bpb']
    best_row = summary.loc[summary['mean_val_bpb'].idxmin()]
    lines = [
        f"best_config\t{best_row['config']}",
        f"best_mean_val_bpb\t{best_row['mean_val_bpb']:.6f}",
    ]
    if not tied.empty:
        lines.append(f"delta_vs_tied\t{best_row['mean_val_bpb'] - float(tied.iloc[0]):.6f}")
    if not untied.empty:
        lines.append(f"delta_vs_untied\t{best_row['mean_val_bpb'] - float(untied.iloc[0]):.6f}")
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.config}\tmean={row.mean_val_bpb:.6f}\tstd={row.std_val_bpb:.6f}\tlost={row.mean_lost_before:.6f}->{row.mean_lost_after:.6f}\trot={row.mean_rotation_updates:.2f}\tlm={row.mean_lm_head_grad_norm:.6f}\twte={row.mean_wte_grad_norm:.6f}"
        )
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
