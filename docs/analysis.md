# Analysis

## Anchors

This repo uses two public anchors:

- `karpathy/autoresearch` for the reference backbone and training harness style
- `tokenbender/nanogpt-attnres-repro` for the faithful AttnRes baseline behavior

The strong baseline in this repo is **not** a modified register variant. It is the faithful `AttnRes block2 softmax` setup with no registers.

## Public families

The public experiment surface is intentionally small:

- `baseline_off`
- `attnres_block2`
- `projected_deepemb_final`
- `projected_deepemb_final_static`
- `projected_deepemb_all`

## Core claim

The method under study is **projected token memory** added on top of the `AttnRes block2` backbone.

The repo separates three evidence tiers:

1. **Main matrix**
   - same backbone family
   - same training budget
   - same seeds
   - compares `baseline_off`, `attnres_block2`, and `projected_deepemb_final`

2. **Mechanism ablation**
   - same seeds as the main matrix
   - fixed LR
   - compares `attnres_block2`, `projected_deepemb_final_static`, `projected_deepemb_final`, and `projected_deepemb_all`
   - answers two questions:
     - does `projected` beat `static`?
     - is `final_only` at least competitive with `all` on the same cohort?

3. **Fresh-seed follow-up**
   - new seeds only
   - fixed LR
   - compares `attnres_block2`, `projected_deepemb_final`, and `projected_deepemb_all`
   - checks robustness outside the main cohort

## Result files

Read the generated TSV summaries, not hard-coded prose numbers:

- `results/main_results.tsv`
- `results/ablation_results.tsv`
- `results/followup_results.tsv`

Read the raw per-run tables when you need per-seed or per-LR detail:

- `results/raw/main_fair_matrix.tsv`
- `results/raw/ablation_matrix.tsv`
- `results/raw/followup_results.tsv`

## Current interpretation

The intended public interpretation is:

- `AttnRes block2` is the strong baseline
- projected token memory is the innovation
- `final_only` is the canonical public form unless same-cohort evidence later proves `all` is materially better

One caveat must remain explicit:

- if `attnres_final_register` stays very high in the winning runs, the method should be described as a **final-path token-memory readout on top of AttnRes**, not as a mild all-layer routing tweak

## Public framing

Safe framing:

- strong baseline: faithful AttnRes block2
- innovation: projected token memory on top of AttnRes
- cleanest final form: final-only projected token memory

Unsafe framing:

- “all AttnRes injection points are equally important”
- “this is a general all-layer AttnRes routing improvement”

The repo is designed so that the release scripts can regenerate all public claims from the raw tables.
