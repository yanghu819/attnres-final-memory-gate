# autoresearch-attnres-project

A clean AutoDL-first repo for studying how to combine **AttnRes** with **scalable n-gram memory**.

This repo no longer uses the old `projected token memory register` framing. The current work has converged to two concrete lines:

1. **Best practical method**
   - `AttnRes block2`
   - final-only **bounded blend**
   - memory source: **unigram + hashed bigram**
   - best current setting: `4 banks x 1M buckets`

2. **Best organic method**
   - `AttnRes block2`
   - memory does **not** become an extra token/source/branch
   - memory only modulates the final AttnRes mixer’s `q/k/v`
   - best current setting: `modqkv q=0.02, k=0.05, v=0.05`

Core code comparison:
- [`docs/core_vs_baseline.md`](docs/core_vs_baseline.md)

## Current status

### Best practical result
From [`results/ngram_module_ablation_results.tsv`](results/ngram_module_ablation_results.tsv) and [`results/bigram_80pct_refine_results.tsv`](results/bigram_80pct_refine_results.tsv):

- `AttnRes block2` strong baseline: `2.154041`
- `unigram + bigram, 4 banks x 1M buckets`: `1.840970`

This is currently the strongest method in the repo.

### Best organic result
From [`results/modqkv_refine_results.tsv`](results/modqkv_refine_results.tsv):

- best `modqkv`: `2.004640`

This is clearly better than the AttnRes baseline, but still worse than the bounded-blend method.

## What actually worked

The key empirical lesson is narrow:

- memory helps most when used **late**
- memory should be **bounded**
- memory should provide **content**
- memory should **not** compete directly with real depth sources in the same softmax

In practice, the strongest version today is:

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

where:

- `y_attnres` is the final output of faithful `AttnRes block2`
- `E_uni` is unigram memory
- `E_bi_b` are independent hashed bigram memory banks
- `g` is a bounded static gate

## What did not work as well

These routes were tested and are not the current main line:

- memory as extra source / KV inside the final softmax
- memory as query token
- pure tied-embedding value residual
- dense scaling past `4 banks x 1M buckets`

The main failure mode was not “memory is ignored”. It was the opposite:

- memory was too easy to attend to
- or memory-induced query shifts collapsed the depth routing

## Repo layout

- `src/autoresearch_attnres_project/legacy_engine.py`
  - core training engine and current memory integration logic
- `src/autoresearch_attnres_project/presets.py`
  - named presets
- `src/autoresearch_attnres_project/metrics.py`
  - log parsing helpers
- `scripts/`
  - AutoDL training, sweep, pull, plot, and analysis scripts
- `results/`
  - checked-in experiment summaries and figures
- `docs/blog.zh.md`
  - Chinese blog post
- `docs/blog.en.md`
  - English blog post

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

### Reproduce the strongest practical line

```bash
bash scripts/repro_ngram_module_ablation.sh
bash scripts/repro_bigram_80pct_refine.sh
```

Useful outputs:

- [`results/ngram_module_ablation_results.tsv`](results/ngram_module_ablation_results.tsv)
- [`results/bigram_80pct_refine_results.tsv`](results/bigram_80pct_refine_results.tsv)
- [`results/figs/fig_ngram_module_ablation_loss_curves.png`](results/figs/fig_ngram_module_ablation_loss_curves.png)
- [`results/figs/fig_bigram_80pct_refine_loss_curves.png`](results/figs/fig_bigram_80pct_refine_loss_curves.png)

### Reproduce the best organic line

```bash
bash scripts/repro_modqkv_refine.sh
```

Useful outputs:

- [`results/modqkv_refine_results.tsv`](results/modqkv_refine_results.tsv)
- [`results/modqkv_refine_analysis.txt`](results/modqkv_refine_analysis.txt)
- [`results/figs/fig_modqkv_refine_loss_curves.png`](results/figs/fig_modqkv_refine_loss_curves.png)
- [`results/figs/fig_modqkv_refine_summary.png`](results/figs/fig_modqkv_refine_summary.png)

## Framing

Use the repo with this framing:

- baseline: faithful `AttnRes block2`
- strongest method: **bounded unigram+bigram final-memory blend**
- best organic method: **memory-conditioned `modqkv`**

Do not frame this repo as a broad all-layer AttnRes routing rewrite. Current evidence supports a narrower claim:

> scalable n-gram memory is most effective on top of AttnRes when used as a bounded final readout, and the cleanest internal integration found so far is small `q/k/v` modulation.
