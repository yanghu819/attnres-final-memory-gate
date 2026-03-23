# autoresearch-attnres-project

A clean AutoDL-first repo for studying how to combine **AttnRes** with **scalable n-gram memory**.

## Current active lines

This repo now keeps only the current active experiment families at the top level.
Historical lines are still preserved under `scripts/archive/` and `results/archive/`.

### 1. Strongest overall method
- backbone: faithful `AttnRes block2`
- usage: final-only bounded blend
- memory: unigram + hashed bigram
- best current setting: `4 banks x 1M buckets`
- best checked-in result: `1.840970`

Core form:

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

### 1b. Best scalable final-blend variant
- same final-only bounded blend as the strongest overall method
- replace the full-dim bigram table with a low-rank hashed bigram code plus a shared projection
- best current setting: `rank=96`, `4 banks x 256k buckets`
- checked-in result: `1.848812`
- mean peak VRAM: `3991.9 MB` versus `15749.0 MB` for the full bigram blend

Core form:

```python
m = W_tied[token] + P_bi(E_bi_code[hash(prev_token, token)])
y = (1 - g) * y_attnres + g * m
```

### 2. Best input-side method
- inspired by `parameter-golf`
- input-side hashed bigram residual + light smear gate
- best checked-in result: `1.969269`
- best setting: `input_bigram_smear_s025`

This is the strongest "small local prior" line, but it still underperforms the final bounded blend.

### 3. Best organic internal method
- memory does not become an extra token, source, or branch
- memory only modulates the final AttnRes mixer's `q/k/v`
- best checked-in result: `2.004640`
- best setting: `modqkv q=0.02, k=0.05, v=0.05`

This is the cleanest internal integration found so far, but it still underperforms the bounded final blend.

## What actually held up

The stable empirical rules are narrow:

- memory helps most when used **late** or **lightly**
- memory should be **bounded** if used as a final readout
- memory should provide **content / prior**, not become a hard competitor inside the same softmax
- input-side local memory can help, but only as a **small residual prior**

## What is no longer current

These lines are preserved only as historical exploration under `archive/`:

- projected token-register family
- unified memory source / query-token routes
- pure tied value-residual route
- LM-head rotation line
- earlier simple-gate / capped-gate / tied-memory / grouped-memory sweeps

They are not the current recommended starting points.

## Repo layout

- `src/autoresearch_attnres_project/legacy_engine.py`
  - core training engine and current memory integration logic
- `src/autoresearch_attnres_project/presets.py`
  - named presets
- `scripts/`
  - only current active training / sweep / plot / analysis scripts
- `scripts/archive/`
  - historical experiment scripts kept for reference
- `results/`
  - current active result summaries and figures
- `results/archive/`
  - historical result tables and figures
- `docs/analysis.md`
  - current interpretation and cleanup status
- `docs/blog.en.md`
  - English blog post
- `docs/blog.zh.md`
  - Chinese blog post

## Reproduction

### Environment

Primary target:
- AutoDL single-GPU box
- Python via `/root/miniconda3/bin/python`
- package manager via `python -m uv`

### Install

```bash
/root/miniconda3/bin/python -m pip install uv
/root/miniconda3/bin/python -m uv sync
```

### Prepare data

```bash
bash scripts/prepare_autodl.sh
```

### Reproduce strongest overall line

```bash
bash scripts/repro_ngram_module_ablation.sh
bash scripts/repro_bigram_80pct_refine.sh
```

Key files:
- `results/ngram_module_ablation_results.tsv`
- `results/bigram_80pct_refine_results.tsv`
- `results/figs/fig_ngram_module_ablation_loss_curves.png`
- `results/figs/fig_bigram_80pct_refine_loss_curves.png`

### Reproduce best scalable final-blend line

```bash
bash scripts/repro_lightweight_final_blend.sh
```

Key files:
- `results/raw/lightweight_final_blend_results.tsv`
- `results/lightweight_final_blend_results.tsv`
- `results/lightweight_final_blend_analysis.txt`
- `results/figs/fig_lightweight_final_blend_valbpb.png`

### Reproduce best organic line

```bash
bash scripts/repro_modqkv_refine.sh
```

Key files:
- `results/modqkv_refine_results.tsv`
- `results/modqkv_refine_analysis.txt`
- `results/figs/fig_modqkv_refine_loss_curves.png`
- `results/figs/fig_modqkv_refine_summary.png`

### Reproduce current input-side line

```bash
bash scripts/repro_input_memory_ablation.sh
bash scripts/repro_input_memory_refine.sh
```

Key files:
- `results/raw/input_memory_ablation_results.tsv`
- `results/input_memory_refine_results.tsv`
- `results/input_memory_refine_analysis.txt`
- `results/figs/fig_input_memory_refine_valbpb.png`

## Framing

Use the repo with this framing:

- strong baseline: faithful `AttnRes block2`
- strongest method: bounded unigram+bigram final-memory blend
- best scalable final-memory method: tied unigram + low-rank bigram final blend
- strongest input-side line: light input bigram prior with smear
- best organic method: memory-conditioned `modqkv`

Do not frame this repo as a broad all-layer AttnRes routing rewrite. Current evidence supports a narrower claim:

> scalable n-gram memory is most effective on top of AttnRes as either a bounded final readout or a small input-side local prior; the cleanest internal integration found so far is small `q/k/v` modulation.
