# Manufactured inverse: unknown compliant Gaussian fixture

This small CPU experiment extends the exact point-pin model to an uncertain, finite-compliance **material-frame Gaussian fixture**. It is a manufactured verification using two tetrahedra and the same linear implicit-Euler mechanics for truth and hypotheses. It provides no Isaac, population calibration, contact, or generalization evidence.

The four unknowns are Young's modulus, one surface coordinate of the fixture center, isotropic Gaussian width, and finite fixture stiffness. All 15 nodal degrees of freedom remain free. Positive-semidefinite spring and damping matrices distribute the fixture through normalized reference-surface weights. The patch stays attached to material coordinates; its rest anchor is fixed in the world. Density, Poisson ratio, and damping are known. The applied nodal force distribution is fixed across every hypothesis, rather than regenerated from each candidate's unknown parameters.

Compatible nodal motion transports volumetric material Gaussian centers and physical covariances using the tetrahedral deformation gradient. Those covariances describe physical shape; uncertainty is represented separately by weighted material/fixture hypotheses.

## Fixed primary experiment

The design was frozen before solving: independent uniform intrinsic priors on log stiffness, center coordinate, log width, and log fixture stiffness; an off-grid truth; a known 0.02 N pulse; and twelve displacement observations at two nodes and two early times with 20 µm Gaussian noise. A second force direction starts from rest. Tensor Gauss–Legendre orders 3 and 5 integrate the prior with their nonuniform product masses. All hypotheses and both levels are retained.

At the primary 625-point level:

| Coordinate RMSE | Prior mean | Conditioned mean |
| --- | ---: | ---: |
| Future motion under observed force A | 126.69 µm | 91.85 µm |
| Independent reset under force B | 84.54 µm | 95.73 µm |

The second prediction gets worse despite narrower uncertainty. Fixture position, width, and compliance remain ambiguous. More seriously, the nominal 90% material-stiffness interval collapses to the single 4000 Pa quadrature node and excludes the off-grid 4300 Pa truth. The 3→5 predictive means change by only 2.15/1.74 µm for A/B, so apparent prediction stability does not establish a reliable material posterior. The effective weight count of 59.69 describes quadrature concentration, not independent samples or sampler mixing.

The maximum force residual is 2.54×10⁻¹⁵ N and the minimum candidate deformation determinant is 0.999390. Six synthetic tests cover physical prior masses, missing-observation ambiguity, informative-likelihood contraction, equality with the existing mechanics bridge, passive-frame objectivity, compatible Gaussian mass/SPD shape, and sensitivity to every unknown. Mechanical consistency does not resolve the posterior quadrature failure.

## Separate bounded numerical diagnosis

A second protocol deliberately changes **Gauss–Legendre to nested Clenshaw–Curtis quadrature**, orders 5/9/17. It preserves the prior, physical parameters, noise, controls, truth, and exact saved observation. It allows two CPU threads and a 55-second wall alarm, retains every completed or partial level, and permits no retry, adaptive node selection, or preferred-result selection.

Acceptance requires all declared levels and the final adjacent comparison to satisfy every marginal CDF, physical quantile, predictive mean, and predictive standard-deviation criterion. It additionally requires the nominal material interval to span at least three stiffness nodes. This is a finite numerical diagnostic; passing would not establish calibrated physical intervals or prove continuous-posterior accuracy.

All three secondary levels completed in 42.35 seconds; the declared numerical criterion **failed**.

| Quadrature | Particles | Future-A RMSE | Reset-B RMSE | E nominal 90% interval | Weight ESS |
| --- | ---: | ---: | ---: | --- | ---: |
| Original Gauss–Legendre 5 | 625 | 91.85 µm | 95.73 µm | [4000, 4000] Pa | 59.69 |
| Clenshaw–Curtis 5 | 625 | 90.21 µm | 96.00 µm | [4000, 4000] Pa | 23.74 |
| Clenshaw–Curtis 9 | 6561 | 93.48 µm | 94.70 µm | [4000, 4000] Pa | 202.03 |
| Clenshaw–Curtis 17 | 83521 | 92.14 µm | 91.28 µm | [3494, 4579] Pa | 3828.45 |

At the final refinement, marginal CDF distances for E, center, width, and fixture stiffness are 0.3020, 0.0709, 0.0609, and 0.0555; all exceed the declared 0.05 limit. The E quantiles move by 9.65% of its physical prior span, exceeding the 1% limit. The interval now spans three E nodes and contains the synthetic truth, but its resolution criterion still fails. Future-A moment refinement passes; reset-B mean-field change is 4.25 µm against a 3.32 µm tolerance. No level is presented as a converged continuous posterior or calibrated physics. No further sweep or retuning was performed.

Four additional analytic tests verify nested positive quadrature masses, uniform-prior polynomial moments, CDF and collapsed-interval detection, preservation of nonuniform prior weights under constant likelihood, and rejection of predictive agreement as a substitute for material resolution.

## Reproduction and scope

The source and frozen protocols are separate from current campaign sources. Raw observations, posterior weights, and moment fields remain local.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q tests/test_campaign_compliant_patch_inverse.py \
  tests/test_diagnose_compliant_patch_resolution.py
```

Primary source: `experiments/campaign_compliant_patch_inverse.py`; design: `runs/operator-dev/compliant-patch-inverse-v01/design.json`; result: `runs/gaussian-campaign-v01/compliant-patch-inverse-v01/summary.json`. Secondary source: `experiments/diagnose_compliant_patch_resolution.py`; protocol: `runs/operator-dev/compliant-patch-resolution-v01/protocol.json`.

To repeat either solve, use a **fresh output directory** and the recorded design/protocol SHA:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python experiments/campaign_compliant_patch_inverse.py run \
  --design runs/operator-dev/compliant-patch-inverse-v01/design.json \
  --design-sha256 928a6ccabfb460fcf5d8295a9600fa71d15210563c08843353d2e9e6a6034081 \
  --output /tmp/compliant-primary-fresh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python experiments/diagnose_compliant_patch_resolution.py run \
  --protocol runs/operator-dev/compliant-patch-resolution-v01/protocol.json \
  --protocol-sha256 de67923268a95edf70d65837393feb27615658abd08e99fed50fd3447b8c694d \
  --output /tmp/compliant-secondary-fresh
```

The secondary command requires the hash-bound local primary observation archive. It cannot be reproduced from source/text alone without regenerating that exact archive.

The fixture family is one center coordinate, one isotropic width, and scalar finite stiffness with fixed damping. It does not infer arbitrary boundary types, environmental unilateral contact, unknown loads, or full material fields. Linear small-strain kinematics and same-model manufactured truth remain substantial limitations. Separate nonlinear Gaussian mechanics and world-contact primitives are independent work; they are not evidence for this inverse experiment.
