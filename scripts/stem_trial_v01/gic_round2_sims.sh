#!/bin/bash
# GIC simulator runs for round 2: fitted-episode trajectories (appearance driving) and D2b forecasts.
R=/home/bayespc2/vscode/guassianSDE; G=$R/runs/stem-trial-v01/gic; PY=$R/runs/gap-crossmethod-v02/gic-env/bin/python
cd $R/runs/gap-crossmethod-v02/vendor/gic
gpu=$1; shift
run() { [ -f "$2" ] || CUDA_VISIBLE_DEVICES=$gpu STEM_TI_MEM=0.3 STEM_FORCE_JSON=$3 STEM_FORCE_AUDIT=${2%.npz}.audit.json $PY -u $R/scripts/stem_trial_v01/gic_stem_predict.py $1 $2 $4; }
for job in "$@"; do eval run $job; done
