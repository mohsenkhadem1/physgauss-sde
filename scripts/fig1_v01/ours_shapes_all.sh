#!/bin/bash
# Full-field shape predictions of ours for the Figure-1 shape rows (sealed with the other predictions).
set -eu
R=/home/bayespc2/vscode/guassianSDE; PY=$R/.venv/bin/python; S=$R/scripts/stem_trial_v01; F=$R/scripts/fig1_v01
for w in W500 W1000 W2000; do FIELDS_ROUND=round3 $PY $S/ours_fields.py rgb $w D2c; done
for w in Bleft Bcentre Bright; do $PY $F/ours_block.py field $w rgb; done
for w in W500 W1000 W2000; do for st in D0 D0+D1; do $PY $S/ours_striker_field.py rgb $w $st; done; done
echo all shapes done
