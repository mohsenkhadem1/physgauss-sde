#!/bin/bash
# GIC striker (D3) forecasts from the passive fit, each refit and the oracle parameters.
R=/home/bayespc2/vscode/guassianSDE; G=$R/runs/stem-trial-v01/gic; PY=$R/runs/gap-crossmethod-v02/gic-env/bin/python
cd $R/runs/gap-crossmethod-v02/vendor/gic
for m in D0 D1-W500 D1-W1000 D1-W2000 oracle-W500 oracle-W1000 oracle-W2000; do
  [ -f $G/forecast-D3-$m.npz ] || STEM_TI_MEM=0.12 STEM_STRIKER_JSON=$G/striker-D3.json $PY -u $R/scripts/stem_trial_v01/gic_stem_predict.py $G/$m $G/forecast-D3-$m.npz 37
done
