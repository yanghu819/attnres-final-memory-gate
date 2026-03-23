#!/usr/bin/env bash
set -euo pipefail

ssh autodl 'bash -s' <<'EOF'
set -euo pipefail
cd /root/autodl-tmp/work/autoresearch-attnres-project
bash scripts/repro_lightweight_final_blend.sh > runs/lightweight_final_blend.queue.log 2>&1
EOF
