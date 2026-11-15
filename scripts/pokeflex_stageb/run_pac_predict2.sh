#!/bin/bash
# usage: run_pac_predict.sh SHARD OBJ [OBJ ...]   PAC-NeRF held-out predictions (shard i/n), waiting for each object's fit
cd /home/bayespc2/vscode/guassianSDE
shard=$1; shift
for o in "$@"; do
  until [ -f runs/pokeflex-v01/stageB/work/pacnerf/$o/fit.json ]; do sleep 30; done
  echo "=== predict $o $shard $(date -Is)"
  scripts/stem_trial_v01/pac_env.sh -I scripts/pokeflex_stageb/pfx_pacnerf.py predict "$o" "$shard" 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "Forward\]\|Backward\]\|velocity: \[\|^\s*$\|it/s\]"
done
echo "=== done $(date -Is)"
