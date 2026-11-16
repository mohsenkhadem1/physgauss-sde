#!/bin/bash
# Usage: probe_chain.sh WORLD SCALE X1 [X2 ...]  -> truth + RGB obs for calibrated presses P<x> (known command)
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python
w=$1; sc=$2; shift 2
for x in "$@"; do
  $P scripts/patch_roles_v02/truth.py "P$x" "$w" "$sc"
  tag=$($P -c "print(f'{float(\"$sc\"):g}')")
  $P scripts/patch_roles_v02/obs.py "P$x" "$w-a$tag"
done
