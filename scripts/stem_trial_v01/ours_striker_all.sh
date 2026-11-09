#!/bin/bash
cd /home/bayespc2/vscode/guassianSDE
for w in W500 W1000 W2000; do .venv/bin/python scripts/stem_trial_v01/ours_striker.py rgb $w D0+D1; done
for w in W500 W1000 W2000; do .venv/bin/python scripts/stem_trial_v01/ours_striker.py rgb $w D0; done
