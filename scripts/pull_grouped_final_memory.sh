#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs/grouped_final_memory results/figs
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/raw/grouped_final_memory_results.tsv \
  results/raw/grouped_final_memory_results.tsv
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/grouped_final_memory/ \
  runs/grouped_final_memory/
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/figs/fig_grouped_final_memory_loss_curves.png \
  results/figs/fig_grouped_final_memory_loss_curves.png || true
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/figs/fig_grouped_final_memory_valbpb.png \
  results/figs/fig_grouped_final_memory_valbpb.png || true
