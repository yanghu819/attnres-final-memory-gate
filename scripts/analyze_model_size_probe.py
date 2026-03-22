from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/model_size_probe_results.tsv'
OUT = ROOT / 'results/model_size_probe_analysis.txt'
SELECTED = ROOT / 'results/model_size_selected_sizes.tsv'

SIZE_ORDER = ['s384_d6', 'm512_d8', 'l640_d10']
FAMILY_ORDER = [
    'strong_baseline',
    'blend_ngram_ref',
    'unified_ngram_projected',
    'query_bigram_only_projected',
    'query_ngram_projected',
]
VRAM_LIMIT_MB = 27000.0


def main() -> None:
    df = pd.read_csv(RAW, sep='\t')
    df['peak_vram_mb'] = pd.to_numeric(df['peak_vram_mb'], errors='coerce')
    df['val_bpb'] = pd.to_numeric(df['val_bpb'], errors='coerce')

    lines: list[str] = []
    selected_rows: list[dict[str, object]] = []
    for size_tag in SIZE_ORDER:
        sub = df[df['size_tag'] == size_tag].copy()
        if sub.empty:
            continue
        max_nonbaseline = sub.loc[sub['config'] != 'strong_baseline', 'peak_vram_mb'].max()
        ok = pd.notna(max_nonbaseline) and float(max_nonbaseline) <= VRAM_LIMIT_MB
        depth = int(sub['depth'].iloc[0])
        head_dim = int(sub['head_dim'].iloc[0])
        dbs = int(sub['device_batch_size'].iloc[0])
        tbs = int(sub['total_batch_size'].iloc[0])
        lines.append(f'{size_tag}: depth={depth} head_dim={head_dim} device_bs={dbs} total_bs={tbs} max_nonbaseline_vram={max_nonbaseline:.1f} fits_80pct={ok}')
        for family in FAMILY_ORDER:
            fam = sub[sub['config'] == family]
            if fam.empty:
                continue
            r = fam.iloc[0]
            lines.append(f'  - {family}: val_bpb={r.val_bpb:.6f} vram={r.peak_vram_mb:.1f}')
        if ok:
            selected_rows.append({
                'size_tag': size_tag,
                'depth': depth,
                'head_dim': head_dim,
                'device_batch_size': dbs,
                'total_batch_size': tbs,
            })

    if not selected_rows:
        raise SystemExit('No feasible size selected from probe.')

    pd.DataFrame(selected_rows).to_csv(SELECTED, sep='\t', index=False)
    OUT.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
