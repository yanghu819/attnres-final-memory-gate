# Analysis

## Anchors

This repo uses two public anchors:

- `karpathy/autoresearch` for the reference backbone and training-harness style
- `tokenbender/nanogpt-attnres-repro` for faithful AttnRes baseline behavior

The strong baseline in this repo is the faithful `AttnRes block2 softmax` setup with no extra registers.

## What changed

Early versions of this repo studied projected token-memory registers. That is no longer the main framing.

The current experimental picture is simpler:

1. the strongest method is a **bounded final blend** between AttnRes output and scalable n-gram memory
2. the best internal or “organic” integration so far is **memory-conditioned `modqkv`** at the final AttnRes mixer

## Current practical method

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

This line consistently beats the AttnRes baseline in the checked-in result tables.

## Current organic method

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

Best current setting in the repo:

- `q = 0.02`
- `k = 0.05`
- `v = 0.05`

This line is clearly positive versus the AttnRes baseline, but it still underperforms the bounded final blend.

## What did not work

These routes were tested and are not the current main line:

- memory as a unified extra source / KV inside the final softmax
- memory as a query token
- pure tied-embedding value residual
- dense scaling beyond the `4 banks x 1M` regime

The key failure modes were:

- memory takeover: memory absorbs too much final-mixer mass
- routing collapse: memory-conditioned query shifts push final routing into degenerate distributions

## Result files

For current strongest practical results:

- `results/ngram_module_ablation_results.tsv`
- `results/bigram_80pct_refine_results.tsv`

For current best organic results:

- `results/modqkv_refine_results.tsv`
- `results/modqkv_refine_analysis.txt`

For source / mechanism analysis:

- `results/unified_memory_tokens_results.tsv`
- `results/unified_memory_tokens_analysis.txt`
- `results/tied_refine_results.tsv`

## Current interpretation

The safe interpretation is now:

- scalable n-gram memory is useful on top of AttnRes
- it works best as a **bounded final readout**
- the cleanest internal integration found so far is **small `q/k/v` modulation**
- broad all-layer routing claims are not supported by the current evidence

## Public framing

Safe framing:

- strong baseline: faithful `AttnRes block2`
- strongest method: bounded unigram+bigram final-memory blend
- best organic method: memory-conditioned `modqkv`

Unsafe framing:

- “we found a uniformly better all-layer AttnRes routing mechanism”
- “projected token register is still the canonical method”
- “memory should compete directly with depth sources in the same softmax”
