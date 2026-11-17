#!/bin/bash
# Usage: step3_after.sh WAIT_UNIT NEW_UNIT GPU_UUID LOG "ARGS1" ...  (step3_run batch after WAIT_UNIT ends)
w=$1; u=$2; g=$3; log=$4; shift 4
cd /home/bayespc2/vscode/guassianSDE
while systemctl --user is-active -q "$w"; do sleep 30; done
scripts/stem_trial_v01/run_unit.sh "$u" 24G $PWD "$log" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$g" -- scripts/patch_roles_v02/step3_batch.sh scripts.patch_roles_v02.step3_run "$@"
