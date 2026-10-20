#!/usr/bin/env bash
set -eu
project_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec bash "$project_root/experiments/launch_isaac_block.sh" \
    --youngs-modulus 1000 --fixture-nodes 20 --load-patch-nodes 9 \
    --force-scale .35 --hz 480 "$@"
