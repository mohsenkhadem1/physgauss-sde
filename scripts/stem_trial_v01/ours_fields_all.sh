#!/bin/bash
cd /home/bayespc2/vscode/guassianSDE
for ep in D2 D2b; do for s in tracks rgbd; do .venv/bin/python scripts/stem_trial_v01/ours_fields.py $s $1 $ep; done; done
