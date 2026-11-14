#!/bin/bash
# PhysTwin Figure-1 pipeline for one world (run as one capped unit, see README):
#   D0 fit (native CMA 20 gens + native first-order 200 it) -> D1 refit initialised from D0 -> forecasts.
# Usage: pipeline.sh SCENE WORLD [TAG]   (TAG suffixes output names, default none = production)
set -euo pipefail
scene=$1; world=$2; tag=${3:-}
cd "$(dirname "$0")/../../.."
P=runs/phystwin-env/bin/python; D=scripts/fig1_v01/phystwin
if [ "$scene" = block ]; then targets="D2"; else targets="D2c D3"; fi
$P $D/fit.py "$scene" "$world" D0 --tag="$tag"
$P $D/fit.py "$scene" "$world" D1 --tag="$tag"
for fe in D0 D1; do $P $D/forecast.py "$scene" "$world" $fe --targets $targets --tag="$tag"; done
echo PIPELINE_DONE "$scene" "$world"
