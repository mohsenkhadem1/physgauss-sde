#!/bin/bash
# Follow-up: re-run probe stages (cached initial SMC) and inverse-crime controls with coarse truth.
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python
for s in iii_a_support_truth iii_b_force_truth ii_h_scale; do echo "=== $s rerun $(date)"; $P -m scripts.patch_roles_v01.run_scene --scene $s --seed 0 || echo "FAILED $s"; done
for s in i_published iii_b_force_truth iv_decoy ii_two_forces; do echo "=== $s coarse truth $(date)"; $P -m scripts.patch_roles_v01.run_scene --scene $s --seed 0 --truth-cells 8 4 4 --no-probe || echo "FAILED $s"; done
echo "=== batch done $(date)"
