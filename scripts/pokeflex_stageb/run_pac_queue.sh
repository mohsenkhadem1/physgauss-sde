#!/bin/bash
# usage: run_pac_queue.sh CMD OBJ [OBJ ...]   (PAC-NeRF adapter CMD = fit | predict for each object in turn)
cd /home/bayespc2/vscode/guassianSDE
cmd=$1; shift
for o in "$@"; do
  echo "=== $cmd $o $(date -Is)"
  scripts/stem_trial_v01/pac_env.sh -I scripts/pokeflex_stageb/pfx_pacnerf.py $cmd "$o" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "Training static\]\|Forward\]\|Backward\]\|velocity: \[\|^\s*$\|Module \|it/s\]"
done
echo "=== done $(date -Is)"
