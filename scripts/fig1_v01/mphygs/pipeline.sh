#!/bin/bash
# Full M-PhyGs protocol for one scene/world: synthesise inputs, D0 fit (curriculum), D1 refit initialised from
# the final D0 material, forecasts of the held-out command(s) from both materials.
# Usage: pipeline.sh SCENE WORLD            (run it inside a capped unit, see README)
# Env:   STAGES_D0 / STAGES_D1  frame curriculum (defaults below), ITERS (default: native 150 per stage)
set -euo pipefail
scene=$1; world=$2
R=/home/bayespc2/vscode/guassianSDE
PY=$R/runs/gap-crossmethod-v02/mphygs-env/bin/python
export LD_LIBRARY_PATH=$R/runs/fig1-v01/tcnn-linklib:/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
S=$R/scripts/fig1_v01/mphygs
cd $R
if [ "$scene" = block ]; then D0=${STAGES_D0:-"12 24"}; D1=${STAGES_D1:-"12 24"}; TG="D2"; E0=3e4
else D0=${STAGES_D0:-"12 24"}; D1=${STAGES_D1:-"12 24 36"}; TG="D2c D3"; E0=4e5; fi  # E0 = PAC-NeRF initial modulus
it=${ITERS:+--iters $ITERS}
ck() { echo "$R/runs/fig1-v01/mphygs/$scene/$1-$world/run/${scene}_$1_$world/alternating_photo_opt_$2frames_bs$2_grid75_uniform_gravity/tracking_00_mat/chunk_00000/material_field.ckpt"; }
for ep in D0 D1; do [ -f runs/fig1-v01/mphygs/$scene/$ep-$world/config.json ] || $PY $S/synth.py $scene $ep $world; done
prev=""
for n in $D0; do
  [ -f "$(ck D0 $n)" ] || $PY $S/fit.py $scene D0 $world --frames $n $it $( [ -n "$prev" ] && echo "--init $prev" || echo "--init-E $E0" ); prev=$(ck D0 $n); done
d0=$prev
for n in $D1; do
  [ -f "$(ck D1 $n)" ] || $PY $S/fit.py $scene D1 $world --frames $n $it --init $prev; prev=$(ck D1 $n); done
d1=$prev
$PY $S/forecast.py $scene $world D0 --ckpt $d0 --targets $TG
$PY $S/forecast.py $scene $world D1 --ckpt $d1 --targets $TG
echo PIPELINE_DONE $scene $world
