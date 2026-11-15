#!/bin/bash
# usage: run_ptw_queue.sh OBJ [OBJ ...]   (fit + predict PhysTwin for each object in turn)
cd /home/bayespc2/vscode/guassianSDE
for o in "$@"; do
  echo "=== $o $(date -Is)"
  runs/phystwin-env/bin/python -I scripts/pokeflex_stageb/pfx_phystwin.py all "$o" 2>&1 | grep --line-buffered -v "^Module " || echo "FAILED $o"
done
echo "=== done $(date -Is)"
