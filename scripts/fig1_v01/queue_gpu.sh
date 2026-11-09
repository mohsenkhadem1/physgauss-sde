#!/bin/bash
# Per-GPU job queue for Figure 1. Usage: queue_gpu.sh GPU BLOCK_WORLD|- STEM_WORLD|-
# Each step is skipped when its output exists; waits on dependency files.
set -u
GPU=$1; BW=$2; SW=$3
R=/home/bayespc2/vscode/guassianSDE; S=$R/scripts/stem_trial_v01; F=$R/scripts/fig1_v01
B=$R/runs/fig1-v01/block; SG=$R/runs/stem-trial-v01/gic; SP=$R/runs/stem-trial-v01/pac
V=$R/runs/gap-crossmethod-v02/vendor/gic; GPY=$R/runs/gap-crossmethod-v02/gic-env/bin/python
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
waitf() { while [ ! -s "$1" ]; do sleep 60; done; }
param() { python3 -c "import json,math,sys;d=json.load(open('$1'));m=d.get('mat_params',d);print(math.log10(m['E']) if '$2'=='logE' else m['$2'])"; }

if [ "$BW" != "-" ]; then
  # GIC block refit after the calibration press, initialised from its D0 fit
  waitf $B/gic/D0-$BW/0-pred.json
  if [ ! -s $B/gic/D1-$BW/0-pred.json ]; then
    log "GIC block D1 refit $BW"
    $R/.venv/bin/python $F/gic_block_config.py $B/gic/config-D1-$BW.json $(param $B/gic/D0-$BW/0-pred.json logE) $(param $B/gic/D0-$BW/0-pred.json nu)
    (cd $V && STEM_FRAMES=25 STEM_TI_MEM=0.25 STEM_FORCE_JSON=$B/forces/force-D1.json STEM_FORCE_AUDIT=$B/gic/D1-$BW-force-audit.json \
      $GPY -u $S/gic_stem_entry.py -c $B/gic/config-D1-$BW.json -s $B/scenes/D1-$BW -m $B/gic/D1-$BW --reg_scale --reg_alpha)
  fi
  for m in D0 D1; do
    [ -s $B/gic/forecast-D2-$m-$BW.npz ] || (cd $V && STEM_TI_MEM=0.25 STEM_FORCE_JSON=$B/forces/force-D2.json \
      $GPY -u $S/gic_stem_predict.py $B/gic/$m-$BW $B/gic/forecast-D2-$m-$BW.npz 25)
  done
  # PAC-NeRF block: passive-press fit, calibration refit, D2 forecasts
  [ -s $B/pac/D0-$BW/fit.json ] || STEM_SCENE=block STEM_FORCE_JSON=$B/forces/force-D0.json $S/pac_env.sh $S/pacnerf_stem.py fit $B/scenes/D0-$BW $B/pac/D0-$BW
  [ -s $B/pac/D1-$BW/fit.json ] || STEM_SCENE=block STEM_FORCE_JSON=$B/forces/force-D1.json $S/pac_env.sh $S/pacnerf_stem.py fit $B/scenes/D1-$BW $B/pac/D1-$BW \
      $(python3 -c "import json;d=json.load(open('$B/pac/D0-$BW/fit.json'));print(d['E'],d['nu'])")
  for m in D0 D1; do
    [ -s $B/pac/forecast-D2-$m-$BW.npz ] || STEM_SCENE=block STEM_FORCE_JSON=$B/forces/force-D2.json $S/pac_env.sh $S/pacnerf_stem.py forecast $B/pac/$m-$BW $B/pac/forecast-D2-$m-$BW.npz 25
  done
  log "block queue $BW done"
fi

if [ "$SW" != "-" ]; then
  waitf $SP/D0/fit.json
  if [ "$SW" = "W500" ]; then  # passive-fit forecasts once (GPU0)
    [ -s $SP/forecast-D2c-D0.npz ] || STEM_FORCE_JSON=$SG/force-D2c.json $S/pac_env.sh $S/pacnerf_stem.py forecast $SP/D0 $SP/forecast-D2c-D0.npz 37
    [ -s $SP/forecast-D3-D0.npz ] || STEM_STRIKER_JSON=$SG/striker-D3.json $S/pac_env.sh $S/pacnerf_stem.py forecast $SP/D0 $SP/forecast-D3-D0.npz 37
  fi
  [ -s $SP/D1-$SW/fit.json ] || STEM_FORCE_JSON=$SG/force-D1.json $S/pac_env.sh $S/pacnerf_stem.py fit $R/runs/stem-trial-v01/scenes/D1-$SW $SP/D1-$SW \
      $(python3 -c "import json;d=json.load(open('$SP/D0/fit.json'));print(d['E'],d['nu'])")
  [ -s $SP/forecast-D2c-D1-$SW.npz ] || STEM_FORCE_JSON=$SG/force-D2c.json $S/pac_env.sh $S/pacnerf_stem.py forecast $SP/D1-$SW $SP/forecast-D2c-D1-$SW.npz 37
  [ -s $SP/forecast-D3-D1-$SW.npz ] || STEM_STRIKER_JSON=$SG/striker-D3.json $S/pac_env.sh $S/pacnerf_stem.py forecast $SP/D1-$SW $SP/forecast-D3-D1-$SW.npz 37
  log "stem queue $SW done"
fi
