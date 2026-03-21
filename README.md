# autoresearch-attnres-project

A clean AutoDL-first reproduction repo for a projected token-memory method on top of a faithful AttnRes strong baseline.

Core code comparison:
- [`docs/core_vs_baseline.md`](docs/core_vs_baseline.md)

This repo is intentionally narrow. It keeps only the public experiment families needed for the paper/blog:

- `baseline_off`: the reference `autoresearch` backbone with AttnRes disabled
- `attnres_block2`: the faithful AttnRes strong baseline (`block_size=2`, softmax, no registers)
- `projected_deepemb_final`: the canonical public method, implemented as projected token memory on top of `attnres_block2`, with `final_only` routing
- `projected_deepemb_all`: an ablation family where the same projected token memory is enabled at all AttnRes injection points
- `projected_deepemb_final_static`: a same-cohort ablation where the token register keeps a static bias instead of a projected bias

## Scope

This repo is for:

- reproducing the clean 3-way main result
- reproducing the same-cohort mechanism ablation
- reproducing the fresh-seed robustness check
- generating the shipped paper/blog figures from raw TSV tables

This repo is not a full lab notebook. Historical branches such as `permix`, `diffv2`, `sigmoid`, `segattn`, `moda`, `anchorkv`, and other exploratory lines are deliberately omitted from the public surface.

## Repo layout

- `src/autoresearch_attnres_project/presets.py`: named public experiment families
- `src/autoresearch_attnres_project/cli.py`: preset-driven training entrypoint
- `src/autoresearch_attnres_project/metrics.py`: log parsing and summary helpers
- `src/autoresearch_attnres_project/legacy_engine.py`: internal training engine ported from `autoresearch`
- `train.py`: thin compatibility wrapper into the packaged engine
- `scripts/repro_main.sh`: canonical 3-way main matrix
- `scripts/repro_ablation.sh`: same-cohort mechanism ablation
- `scripts/repro_followup.sh`: fresh-seed robustness check
- `scripts/repro_release.sh`: runs all three matrices and regenerates plots

## Environment

Primary target:

- AutoDL single-GPU box
- Python via `/root/miniconda3/bin/python`
- package manager via `python -m uv`

## Quick start

### 1. Install dependencies

```bash
/root/miniconda3/bin/python -m pip install uv
/root/miniconda3/bin/python -m uv sync
```

### 2. Prepare data

```bash
bash scripts/prepare_autodl.sh
```

### 3. Inspect the shipped presets

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m autoresearch_attnres_project.cli list-presets
```

### 4. Reproduce the main result

```bash
bash scripts/repro_main.sh
/root/miniconda3/bin/python scripts/plot_main.py
```

Outputs:

- `results/raw/main_fair_matrix.tsv`
- `results/main_results.tsv`
- `results/figs/fig_main_loss_curves.png`
- `results/figs/fig_main_valbpb_by_lr.png`

### 5. Reproduce the mechanism ablation

```bash
bash scripts/repro_ablation.sh
/root/miniconda3/bin/python scripts/plot_ablation.py
```

Outputs:

- `results/raw/ablation_matrix.tsv`
- `results/ablation_results.tsv`
- `results/figs/fig_ablation_loss_curves.png`
- `results/figs/fig_ablation_valbpb.png`

### 6. Reproduce the fresh-seed follow-up

```bash
bash scripts/repro_followup.sh
/root/miniconda3/bin/python scripts/plot_followup.py
```

Outputs:

- `results/raw/followup_results.tsv`
- `results/followup_results.tsv`
- `results/figs/fig_followup_loss_curves.png`
- `results/figs/fig_followup_delta_vs_control.png`
- `results/figs/fig_followup_valbpb.png`

### 7. Run the full release bundle

```bash
bash scripts/repro_release.sh
```

## Preset families

### `baseline_off`
- `attnres_mode=off`

### `attnres_block2`
- `attnres_mode=block`
- `attnres_block_size=2`
- `attnres_weight_mode=softmax`
- `attnres_num_registers=0`
- `attnres_token_registers=0`

### `projected_deepemb_final`
- `attnres_mode=block`
- `attnres_block_size=2`
- `attnres_weight_mode=softmax`
- `attnres_num_registers=0`
- `attnres_token_registers=1`
- `attnres_token_register_mode=query_local`
- `attnres_token_register_bias_mode=projected`
- `attnres_token_register_value_mode=rmsnorm`
- `attnres_register_scope=final_only`
- `attnres_token_register_scale=0.3125`

### `projected_deepemb_final_static`
- same as `projected_deepemb_final`, but `attnres_token_register_bias_mode=static`

### `projected_deepemb_all`
- same as `projected_deepemb_final`, but `attnres_register_scope=all`
- used as an ablation family, not as the canonical public method

## Public framing

Use this repo with the following framing:

- `baseline_off`: raw `autoresearch` baseline
- `attnres_block2`: faithful AttnRes strong baseline
- `projected_deepemb_final`: our canonical method

The mechanism should be described carefully. Current evidence supports:

- projected token memory improves the faithful AttnRes strong baseline
- the final mixer carries most of the gain
- the cleanest public form is `final_only`

Avoid claiming a uniform all-layer AttnRes improvement unless a future rerun explicitly supports that.

## Read first

- `docs/analysis.md`
- `docs/blog.en.md`
- `docs/blog.zh.md`
- `results/main_results.tsv`
- `results/ablation_results.tsv`
- `results/followup_results.tsv`
