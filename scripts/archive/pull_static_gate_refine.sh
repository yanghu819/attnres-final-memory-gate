#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs/static_gate_refine results/figs
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/raw/static_gate_refine_results.tsv \
  results/raw/static_gate_refine_results.tsv
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/static_gate_refine/ \
  runs/static_gate_refine/
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/figs/fig_static_gate_refine_loss_curves.png \
  results/figs/fig_static_gate_refine_loss_curves.png || true
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/figs/fig_static_gate_refine_valbpb.png \
  results/figs/fig_static_gate_refine_valbpb.png || true
