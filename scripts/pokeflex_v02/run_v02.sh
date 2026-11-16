#!/bin/bash
# Launch per-object v02 fit + held-out prediction as capped systemd user units (1 thread each, 8 objects).
# usage: scripts/pokeflex_v02/run_v02.sh [OBJ ...]
cd "$(dirname "$0")/../.." || exit 1
R=$PWD; PY=$R/.venv/bin/python; mkdir -p runs/pokeflex-v02/logs
objs=${*:-FoamDice FoamCylinder Sponge MemoryFoam FoamHalfSphere Pillow 3dPrintedCylinder 3dPrintedPyramid}
for o in $objs; do
  scripts/stem_trial_v01/run_unit.sh pfx2-fit-$o 10G "$R" "$R/runs/pokeflex-v02/logs/fit-$o.log" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 -- \
    /bin/bash -c "$PY -I scripts/pokeflex_v02/v02_fit.py fit $o && $PY -I scripts/pokeflex_v02/v02_fit.py predict $o"
done
