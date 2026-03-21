#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_autodl_env.sh" train --preset attnres_block2 "$@"
