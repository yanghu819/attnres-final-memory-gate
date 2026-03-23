#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="/root/autodl-tmp/work/autoresearch-attnres-project"

mkdir -p "$ROOT/results/raw" "$ROOT/results/figs" "$ROOT/runs"

scp autodl:"$REMOTE_ROOT/results/raw/memory_sources_results.tsv" "$ROOT/results/raw/" || true
scp autodl:"$REMOTE_ROOT/results/memory_sources_results.tsv" "$ROOT/results/" || true
scp autodl:"$REMOTE_ROOT/results/figs/fig_memory_sources_loss_curves.png" "$ROOT/results/figs/" || true
scp autodl:"$REMOTE_ROOT/results/figs/fig_memory_sources_valbpb.png" "$ROOT/results/figs/" || true
scp autodl:"$REMOTE_ROOT/runs/memory_sources.queue.log" "$ROOT/runs/" || true
