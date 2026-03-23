# Results Layout

This directory now keeps only the **current active** result summaries at the top level.

## Active top-level results

Current top-level files correspond to the repo's active lines:

- strongest overall bounded final blend
- best scalable final-memory blend
- best input-side local-memory line
- best organic `modqkv` line

Top-level raw files kept active:

- `results/raw/ngram_module_ablation_results.tsv`
- `results/raw/bigram_80pct_refine_results.tsv`
- `results/raw/lightweight_final_blend_results.tsv`
- `results/raw/input_memory_ablation_results.tsv`
- `results/raw/input_memory_refine_results.tsv`
- `results/raw/modqkv_refine_results.tsv`

Top-level figures kept active:

- `results/figs/fig_ngram_module_ablation_loss_curves.png`
- `results/figs/fig_ngram_module_ablation_valbpb.png`
- `results/figs/fig_bigram_80pct_refine_loss_curves.png`
- `results/figs/fig_bigram_80pct_refine_valbpb.png`
- `results/figs/fig_lightweight_final_blend_valbpb.png`
- `results/figs/fig_input_memory_refine_valbpb.png`
- `results/figs/fig_modqkv_refine_loss_curves.png`
- `results/figs/fig_modqkv_refine_summary.png`

## Archived results

Older exploratory lines were moved to:

- `results/archive/`
- `results/archive/raw/`
- `results/archive/figs/`

These archived files are preserved for reference, but they are not the repo's current recommended starting points.
