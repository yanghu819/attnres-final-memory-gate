# Analysis

## Current Bottom Line

This repo has converged to one main result and one small-budget variant.

### Practical Canonical Method

```python
m = W_tied[token] + P_bi(E_bi_code[hash(prev_token, token)])
y = (1 - g) * y_attnres + g * m
```

Properties:
- backbone: faithful `AttnRes block2`
- readout: final-only bounded blend
- memory: tied unigram + low-rank hashed bigram code
- canonical preset: `final_memory_tied_bigram_factorized_r128_b524288_c085`
- refine 45..48 mean: `1.840357`
- verify 49..52 mean: `1.839716`
- mean peak VRAM: `5277.8 MB`

This is the repo's current best practice.

### Full-Capacity Reference

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

Properties:
- same bounded final blend location
- full hidden-dim unigram+bigram memory
- verify 49..52 mean: `1.840078`
- mean peak VRAM: `15749.0 MB`

This is still the strongest full-capacity reference, but it is not the practical default anymore.

## Golf Track Interpretation

Tiny `parameter-golf` style experiments produced a different conclusion.

### Tiny short-budget best
- preset: `golf_final_lightblend_r96_b65536_c090`
- 10-minute broad sweep 45..48 mean: `1.658716`
- artifact budget: within `16MB` at low-bit export

### Tiny holdout result
On the 20-minute holdout:
- `golf_attnres_tiny`: `1.550197`
- `golf_final_lightblend_r96_b65536_c090`: `1.571063`

So the tiny final-lightblend line is currently a **short-budget accelerator**, not a stable longer-budget canonical.

## What Held Up

The stable rules are:

- use memory **late** if you want the largest gain
- make final readout memory **bounded**
- prefer **content/prior** over direct softmax competition
- if you need scalability, compress the bigram memory into a **low-rank code** instead of keeping a full hidden-dim table
- input-side bigram priors only work when they stay **small**

## Secondary Lines

Still useful as comparison points, but not the repo default:

- input local prior: `input_bigram_smear_s025_b64k`
- organic internal method: `modqkv q=0.02, k=0.05, v=0.05`

## Archived Lines

The following are historical exploration rather than current recommendations:

- projected token-register family
- unified memory source / query-token routes
- pure tied value-residual route
- LM-head rotation
- intermediate golf probe / refine scripts and raw tables
