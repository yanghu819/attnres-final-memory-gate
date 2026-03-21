# AutoDL Notes

Target environment:

- GPU: single NVIDIA GPU on AutoDL
- Python: `/root/miniconda3/bin/python`
- package manager: `python -m uv`

## Setup

```bash
cd /root/autodl-tmp/work/autoresearch-attnres-project
/root/miniconda3/bin/python -m pip install uv
/root/miniconda3/bin/python -m uv sync
bash scripts/prepare_autodl.sh
```

## Named presets

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m autoresearch_attnres_project.cli list-presets
```

Single-run examples:

```bash
bash scripts/train_baseline.sh --seed 41 --reference-lr 0.0003
bash scripts/train_attnres.sh --seed 41 --reference-lr 0.0012
bash scripts/train_projected.sh --seed 41 --reference-lr 0.0012
bash scripts/train_projected_static.sh --seed 41 --reference-lr 0.0012
```

## Main reproduction

```bash
bash scripts/repro_main.sh
/root/miniconda3/bin/python scripts/plot_main.py
```

Outputs:

- `results/raw/main_fair_matrix.tsv`
- `results/main_results.tsv`
- `results/figs/fig_main_loss_curves.png`
- `results/figs/fig_main_valbpb_by_lr.png`

## Mechanism ablation

```bash
bash scripts/repro_ablation.sh
/root/miniconda3/bin/python scripts/plot_ablation.py
```

Outputs:

- `results/raw/ablation_matrix.tsv`
- `results/ablation_results.tsv`
- `results/figs/fig_ablation_loss_curves.png`
- `results/figs/fig_ablation_valbpb.png`

## Fresh-seed follow-up

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

## Full release bundle

```bash
bash scripts/repro_release.sh
```

## Canonical public method

Use this as the public paper/blog method:

- `attnres_mode=block`
- `attnres_block_size=2`
- `attnres_weight_mode=softmax`
- `attnres_token_registers=1`
- `attnres_token_register_mode=query_local`
- `attnres_token_register_bias_mode=projected`
- `attnres_token_register_value_mode=rmsnorm`
- `attnres_register_scope=final_only`
- `attnres_token_register_scale=0.3125`

Use `projected_deepemb_all` only as an ablation family.
