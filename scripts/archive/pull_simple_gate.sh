#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw runs/simple_gate
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/results/raw/simple_gate_results.tsv \
  results/raw/simple_gate_results.tsv
rsync -az \
  autodl:/root/autodl-tmp/work/autoresearch-attnres-project/runs/simple_gate/ \
  runs/simple_gate/
