#!/bin/bash
# Seed replicates: SMC Monte Carlo variability of log evidence and role posteriors.
P=/home/bayespc2/vscode/guassianSDE/.venv/bin/python
for s in i_published iii_a_support_truth iii_b_force_truth iv_decoy; do echo "=== $s seed1 $(date)"; $P -m scripts.patch_roles_v01.run_scene --scene $s --seed 1 --no-probe || echo "FAILED $s"; done
echo "=== batch done $(date)"
