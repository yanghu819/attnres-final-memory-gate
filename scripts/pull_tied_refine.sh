#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="/root/autodl-tmp/work/autoresearch-attnres-project"

mkdir -p "$ROOT/results/raw" "$ROOT/results/figs" "$ROOT/runs"

scp autodl:"$REMOTE_ROOT/results/raw/tied_refine_results.tsv" "$ROOT/results/raw/" || true
scp autodl:"$REMOTE_ROOT/results/tied_refine_results.tsv" "$ROOT/results/" || true
scp autodl:"$REMOTE_ROOT/results/figs/fig_tied_refine_loss_curves.png" "$ROOT/results/figs/" || true
scp autodl:"$REMOTE_ROOT/results/figs/fig_tied_refine_valbpb.png" "$ROOT/results/figs/" || true
scp autodl:"$REMOTE_ROOT/runs/tied_refine.queue.log" "$ROOT/runs/" || true
