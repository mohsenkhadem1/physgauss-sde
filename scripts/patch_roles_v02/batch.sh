#!/bin/bash
# Usage: batch.sh "ARGS1" "ARGS2" ...   (each ARGS is a run.py argument string; runs sequentially)
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python
for a in "$@"; do
  echo "=== $a $(date)"
  $P -m scripts.patch_roles_v02.run $a || echo "FAILED $a"
done
echo "=== batch done $(date)"
