#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_autodl_env.sh" train --preset projected_deepemb_final "$@"
