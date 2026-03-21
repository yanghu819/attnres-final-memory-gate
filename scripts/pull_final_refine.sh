#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/raw runs

rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/raw/final_refine_results.tsv \
  results/raw/final_refine_results.tsv

mkdir -p runs
rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/final_refine.queue.log \
  runs/final_refine.queue.log

mkdir -p runs/final_refine
rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/final_refine/ \
  runs/final_refine/

