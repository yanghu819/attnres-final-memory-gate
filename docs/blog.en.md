# N-Gram Memory on Top of AttnRes: Current Best Practice

This post reflects the current state of the experiments. It does not use the older `projected token register` framing.

## The short version

The strongest method we have today is not “put memory tokens into the AttnRes softmax” and it is not “turn memory into the query”.

The strongest method is simpler:

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

That means:

- the backbone is still faithful `AttnRes block2`
- memory is **unigram + hashed bigram**
- memory is blended into the final output through a **bounded gate**

In plain words:

**let AttnRes do the main reasoning first, then let n-gram memory hand it a small cheat sheet at the end.**

## What this memory is

The memory here is closer to an Engram-style hashed memory than to a small ordinary embedding:

- `unigram`: memory indexed by the current token
- `bigram`: memory indexed by `(prev_token, token)`
- `bank`: an independent hashed bigram table

The current best setting uses:

- `4 banks`
- `1M buckets` per bank

So we keep four independent bigram memory tables and add their outputs together.

## Why this works better than more “organic” integrations

We tested the more unified versions directly:

1. memory as an extra source / KV inside the final softmax  
Result: memory gets attended too easily and pushes real depth sources out.

2. memory as a query token  
Result: it pushes the final routing into degenerate distributions, sometimes collapsing to `x0 = 1.0`.

3. memory only modulating the final AttnRes mixer’s `q/k/v`  
Result: this is the best organic version so far, but it still does not beat the bounded blend.

So the main rule right now is:

- use memory **late**
- keep memory **bounded**
- use memory as **content prior**
- do **not** let memory compete directly with real depth sources in the same softmax

## The strongest current result

Relevant tables:

- [`results/ngram_module_ablation_results.tsv`](../results/ngram_module_ablation_results.tsv)
- [`results/bigram_80pct_refine_results.tsv`](../results/bigram_80pct_refine_results.tsv)

Current best practical method:

- `AttnRes block2` baseline: `2.154041`
- `unigram + bigram, 4 banks x 1M buckets`: `1.840970`

That is the current best practice.

## The best organic version

If you insist on avoiding an explicit external blend branch, the best version so far is:

- memory adds no extra token
- memory adds no extra source
- memory only modulates the final AttnRes mixer’s `q/k/v`

That is `modqkv`:

```python
q' = q + 0.02 * Wq(memory)
k' = k + 0.05 * Wk(memory)
v' = v + 0.05 * Wv(memory)
```

Relevant result table:

- [`results/modqkv_refine_results.tsv`](../results/modqkv_refine_results.tsv)

Current best organic result:

- `modqkv q=0.02, k=0.05, v=0.05`: `2.004640`

This is clearly better than the AttnRes baseline, but still worse than the bounded blend.

## One important scaling result

Making the memory bigger did not keep helping forever.

We tested:

- `4 banks x 1M`
- `5 banks x 1M`
- `6 banks x 1M`
- `7 banks x 1M`

Result:

- `4x1m` was best
- pushing larger than that did not keep improving

So the problem is not simply “more memory is always better”.
It is more likely:

**the memory structure has to be right.**

## What should come next

If we keep pushing the current best-practice line, the most sensible next steps are:

1. `unigram + bigram + trigram`
2. `tied unigram + bigram residual + trigram residual`
3. sparse bank selection instead of larger dense bank summation

That means improving the **structure of the memory**, not making the AttnRes routing itself more complicated.

## Summary

The strongest method today is not “a more complex AttnRes”.

It is:

**AttnRes + bounded unigram/bigram memory readout**

And the best organic attempt so far is:

**memory-conditioned `modqkv`**

But at the moment, the plain bounded final blend still wins.
