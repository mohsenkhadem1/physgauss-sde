#!/bin/bash
# Render + compose a list of Figure-1 Blender scenes on one GPU. usage: blender_all.sh GPU_UUID SCEN-WORLD ...
R=/home/bayespc2/vscode/guassianSDE; G=$1; shift
for sw in "$@"; do
  s=${sw%%-*}; w=${sw#*-}
  [ -f $R/runs/fig1-v01/blender/$sw.npz ] || $R/.venv/bin/python $R/scripts/fig1_v01/blender_export.py $s $w || continue
  rm -rf $R/runs/fig1-v01/blender/frames/$sw
  CUDA_VISIBLE_DEVICES=$G /home/bayespc2/apps/blender-4.2.3-linux-x64/blender -b -P $R/scripts/fig1_v01/blender_render.py -- \
    $R/runs/fig1-v01/blender/$sw.npz $R/runs/fig1-v01/blender/frames/$sw > $R/runs/fig1-v01/blender/render-$sw.log 2>&1 || { echo "$sw render FAILED"; continue; }
  $R/.venv/bin/python $R/scripts/fig1_v01/blender_compose.py $s $w && echo "$sw done"
done
