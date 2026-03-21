# Projected Token Memory on Top of AttnRes

## Setup

This project starts from two anchors:

- the `autoresearch` reference backbone
- a faithful AttnRes reproduction

From there, we add a projected token-memory register on top of `AttnRes block2`.

## The comparison we actually care about

There are three levels of baseline in this repo:

1. `baseline_off`
   - raw `autoresearch` backbone
2. `attnres_block2`
   - faithful AttnRes strong baseline
3. `projected_deepemb_final`
   - our canonical method

That is the main 3-way comparison. The exact current numbers live in:

- `results/main_results.tsv`

## What the method is

The method is not presented here as a broad all-layer residual-topology rewrite.

The clean public form is:

- AttnRes `block_size=2`
- one token-conditioned register
- projected bias
- `query_local` token table
- `rmsnorm` value mode
- `final_only` routing

In short:

**AttnRes backbone + projected token memory at the final mixer**

## What the ablation is for

The same-cohort ablation exists to answer two narrow questions:

1. does `projected` beat `static`?
2. does `final_only` stay competitive with `all` on the same seeds?

Read:

- `results/ablation_results.tsv`
- `results/figs/fig_ablation_loss_curves.png`
- `results/figs/fig_ablation_valbpb.png`

## What the fresh-seed follow-up is for

The fresh-seed follow-up is not the main result. It is the robustness check.

Read:

- `results/followup_results.tsv`
- `results/figs/fig_followup_loss_curves.png`
- `results/figs/fig_followup_valbpb.png`

## Important caveat

This repo should be described honestly.

If the winning runs still show very large `attnres_final_register`, then the safest interpretation is:

- projected token memory helps
- the final mixer is the main place where it helps
- this is closer to a final-path token-memory readout on top of AttnRes than to a uniform all-layer routing improvement

That caveat makes the claim narrower, but also more defensible.
