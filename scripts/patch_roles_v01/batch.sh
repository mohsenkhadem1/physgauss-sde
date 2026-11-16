#!/bin/bash
# Usage: batch.sh SEED PARTICLES MOVES scene1 scene2 ...
seed=$1; np=$2; mv=$3; shift 3
for s in "$@"; do
  echo "=== $s seed $seed $(date)"
  /home/bayespc2/vscode/guassianSDE/.venv/bin/python -m scripts.patch_roles_v01.run_scene --scene "$s" --seed "$seed" --particles "$np" --moves "$mv" || echo "FAILED $s"
done
echo "=== batch done $(date)"
