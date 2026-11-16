#!/bin/bash
# Usage: obs_chain.sh EPISODE-WORLD ...   (renders + tracks runs/patch-roles-v02/truth/EPISODE-WORLD-a0.333333.npz)
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python
for e in "$@"; do $P scripts/patch_roles_v02/obs.py "${e%%-*}" "${e#*-}-a0.333333"; done
