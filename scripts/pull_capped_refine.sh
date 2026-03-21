#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/raw runs

rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/raw/capped_refine_results.tsv \
  results/raw/capped_refine_results.tsv

rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/capped_refine.queue.log \
  runs/capped_refine.queue.log

mkdir -p runs/capped_refine
rsync -av \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/capped_refine/ \
  runs/capped_refine/

