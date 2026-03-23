# Analysis

## Anchors

This repo uses two public anchors:

- `karpathy/autoresearch` for the reference backbone and training-harness style
- `tokenbender/nanogpt-attnres-repro` for faithful AttnRes baseline behavior

The strong baseline in this repo is faithful `AttnRes block2 softmax` with no extra registers.

## Current active picture

The repo has been cleaned so that only current active experiment families remain at the top level.
Older lines are preserved under `scripts/archive/` and `results/archive/`.

The current picture is:

1. the strongest method overall is a **bounded final blend** between AttnRes output and scalable unigram+bigram memory
2. the best scalable final-memory variant is the same bounded final blend, but with a **low-rank hashed bigram code** instead of a full hidden-dim bigram table
3. the strongest input-side line is a **small hashed bigram prior** with a light smear gate
4. the best internal or “organic” integration so far is **memory-conditioned `modqkv`** at the final AttnRes mixer

## Strongest overall method

Current strongest practical method:

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

Properties:

- backbone: faithful `AttnRes block2`
- memory: unigram + hashed bigram
- scaling: `4 banks x 1M buckets`
- usage: final-only bounded readout
- best checked-in result: `1.840970`

This remains the strongest line in the repo.

## Best scalable final-memory variant

Current best scalable variant:

```python
m = W_tied[token] + P_bi(E_bi_code[hash(prev_token, token)])
y = (1 - g) * y_attnres + g * m
```

Properties:

- same late bounded final blend as the strongest method
- unigram base reuses the tied token embedding
- bigram memory is stored as a low-rank hashed code and projected once into hidden size
- best checked-in setting: `rank=96`, `4 banks x 256k buckets`
- best checked-in result: `1.848812`
- mean peak VRAM: `3991.9 MB`

This is currently the most practical version of the strongest overall idea. It gives up only about `0.008` bpb versus the full bigram blend while cutting peak VRAM by roughly `4x`.

## Strongest input-side method

Current best input-side method:

```python
x_t = x_t + small_bigram_hash_residual(x_{t-1}, x_t)
x_t = smear_gate(x_t, x_{t-1})
```

Properties:

- memory is injected at the input side as a small local prior
- best current setting: `input_bigram_smear_s025`
- best checked-in result: `1.969269`

Important empirical rule:
- input memory only works when injected weakly
- larger input residuals and current trigram variants hurt

## Best organic internal method

Current best organic line:

```python
q' = q + aq * Wq(memory)
k' = k + ak * Wk(memory)
v' = v + av * Wv(memory)
y = AttnResFinal(q', k', v')
```

Properties:

- no extra memory token
- no extra source in the final softmax
- no external blend branch
- memory only modulates the final AttnRes mixer internals
- best current setting: `q = 0.02, k = 0.05, v = 0.05`
- best checked-in result: `2.004640`

This line is clearly positive versus the AttnRes baseline, but it still underperforms the bounded final blend.

## What did not hold up

These routes were tested and are archived rather than active:

- projected token-register family
- memory as a unified extra source / KV inside the final softmax
- memory as a query token
- pure tied-embedding value residual
- LM-head rotation line
- dense memory scaling beyond the `4 banks x 1M` sweet spot
- current trigram input-memory variants

Key failure modes:

- memory takeover: memory absorbs too much final-mixer mass
- routing collapse: memory-conditioned query shifts push final routing into degenerate distributions
- over-strong input priors: local memory becomes a destructive residual instead of a small hint

## Current empirical rules

The safe interpretation is now:

- scalable n-gram memory is useful on top of AttnRes
- it works best as a **bounded final readout**
- input-side local memory is useful only as a **small prior**
- the cleanest internal integration found so far is **small `q/k/v` modulation**
- broad all-layer routing claims are not supported by the current evidence

## Current active result files

Strongest overall:
- `results/ngram_module_ablation_results.tsv`
- `results/bigram_80pct_refine_results.tsv`

Best scalable final-memory variant:
- `results/raw/lightweight_final_blend_results.tsv`
- `results/lightweight_final_blend_results.tsv`
- `results/lightweight_final_blend_analysis.txt`

Best input-side line:
- `results/raw/input_memory_ablation_results.tsv`
- `results/input_memory_refine_results.tsv`
- `results/input_memory_refine_analysis.txt`

Best organic line:
- `results/modqkv_refine_results.tsv`
- `results/modqkv_refine_analysis.txt`

Historical results:
- `results/archive/`

## Public framing

Safe framing:

- strong baseline: faithful `AttnRes block2`
- strongest method: bounded unigram+bigram final-memory blend
- best scalable final-memory variant: tied unigram + low-rank bigram final blend
- strongest input-side line: light input bigram prior with smear
- best organic method: memory-conditioned `modqkv`

Unsafe framing:

- “we found a uniformly better all-layer AttnRes routing mechanism”
- “projected token register is still the canonical method”
- “memory should compete directly with depth sources in the same softmax”
