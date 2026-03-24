# autoresearch-attnres-project

A focused repo for studying how **AttnRes** benefits from **scalable n-gram memory**.

## Recommended Best Practices

### 1. Practical Canonical Method
Use this first.

- backbone: faithful `AttnRes block2`
- readout: **final-only bounded blend**
- memory: **tied unigram + low-rank hashed bigram code**
- canonical preset: `final_memory_tied_bigram_factorized_r128_b524288_c085`
- checked result:
  - refine 45..48 mean: `1.840357`
  - verify 49..52 mean: `1.839716`
- mean peak VRAM: `5277.8 MB`

Core form:

```python
m = W_tied[token] + P_bi(E_bi_code[hash(prev_token, token)])
y = (1 - g) * y_attnres + g * m
```

This is the current repo-wide **best practice** because it keeps essentially full performance while cutting memory cost hard.

### 2. Full-Capacity Reference
Use this when you only care about absolute headroom.

- same final-only bounded blend
- memory: **full hidden-dim unigram + hashed bigram**
- checked result:
  - verify 49..52 mean: `1.840078`
- mean peak VRAM: `15749.0 MB`

Core form:

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

This remains the full-capacity reference, but it is no longer the practical default.

### 3. Golf Short-Budget Best
Use this only for `parameter-golf` style tiny, short-budget experiments.

- tiny backbone: `golf_attnres_tiny`
- method: tiny final bounded blend with factorized bigram code
- canonical preset: `golf_final_lightblend_r96_b65536_c090`
- 10-minute broad sweep mean 45..48: `1.658716`
- artifact budget: within `16MB` at low-bit export

Important limit:
- on the 20-minute holdout, tiny baseline beat the lightblend variants
- so this is a **short-budget weapon**, not a stable long-budget canonical yet

## Secondary Lines

These are still informative, but not the repo's main recommendation.

- input-side local prior: `input_bigram_smear_s025_b64k`
  - useful small prior
  - still weaker than the final bounded blend line
- organic internal integration: `modqkv q=0.02, k=0.05, v=0.05`
  - clean internal method
  - still weaker than the bounded final blend line

## What Actually Held Up

The stable empirical rules are narrow:

- memory helps most when used **late** or **lightly**
- final readout memory must be **bounded**
- memory should act as **content/prior**, not a hard competitor inside the same softmax
- input-side memory only works as a **small local residual prior**
- low-rank bigram codes are enough to keep almost all of the full blend's gain

## Start Here

### Reproduce the practical canonical method

```bash
bash scripts/repro_lightweight_final_blend_refine.sh
bash scripts/repro_lightweight_final_blend_verify.sh
```

Key files:
- `results/raw/lightweight_final_blend_refine_results.tsv`
- `results/raw/lightweight_final_blend_verify_results.tsv`
- `results/lightweight_final_blend_refine_analysis.txt`
- `results/lightweight_final_blend_verify_analysis.txt`
- `results/figs/fig_lightweight_final_blend_refine_valbpb.png`

### Reproduce the golf short-budget line

```bash
bash scripts/repro_golf_8h_final_lightblend.sh
```

Key files:
- `results/raw/golf_8h_final_lightblend_results.tsv`
- `results/golf_8h_final_lightblend_analysis.txt`

## Repo Layout

- `src/autoresearch_attnres_project/legacy_engine.py`
  - core training engine and memory integrations
- `src/autoresearch_attnres_project/presets.py`
  - named presets, including the canonical practical and golf variants
- `scripts/`
  - current active reproduction and analysis scripts
- `scripts/archive/`
  - exploratory or superseded scripts
- `results/`
  - current active summaries and figures
- `results/archive/`
  - historical and exploratory outputs
- `docs/analysis.md`
  - current interpretation

## Framing

Use this repo with the following framing:

- strongest practical method: **final bounded blend with tied unigram + low-rank bigram code**
- full-capacity reference: **final bounded full unigram+bigram blend**
- golf short-budget method: **tiny final lightblend**

Do not frame this repo as a broad all-layer AttnRes routing rewrite. The current evidence supports a narrower claim:

> AttnRes benefits most from scalable n-gram memory as a bounded final readout; a low-rank hashed bigram code preserves almost all of the gain while cutting memory cost sharply.
