#!/bin/bash
# Usage: step3_batch.sh MODULE "ARGS1" "ARGS2" ...  (sequential runs of python -m MODULE ARGS)
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python; m=$1; shift
for a in "$@"; do echo "=== $a $(date)"; $P -m $m $a || echo "FAILED $a"; done; echo "=== batch done $(date)"
