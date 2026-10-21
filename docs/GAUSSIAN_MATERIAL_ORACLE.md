# Sparse-depth material oracle

The first known-support experiment improves future deformation prediction over
an unconditional physical prior. It uses a fresh material draw, two sparse depth
frames, and a numerical posterior over four material parameters. Supports,
reference geometry, rigid hardware and commanded forces are supplied.

| Target | Unconditional prior mean | Conditioned mean |
|---|---:|---:|
| Program A, future after the observed 1 s prefix | 0.370 mm | 0.113 mm |
| Program B, held force program from a fresh rest state | 0.216 mm | 0.080 mm |

These are coordinate RMSEs against the independent finite-strain FEM trajectory.
The prior comparison is an explicitly **post hoc diagnostic**, with 8,192 fresh
proper-prior draws and mean Monte Carlo errors of about 3–4 µm. No posterior,
training budget or hyperparameter was changed for that comparison. Program B
is a reset-force prediction, not an experiment that pokes the continuing state.
Held-camera depth errors of the conditioned mean surface are 0.166 and 0.105 mm
for A and B, with no visibility disagreement in this example.

This is one known-support oracle with an analytic depth-ray sensor. Unknown
boundary inference, an RGB encoder, physical uncertainty calibration across
independent objects and active probing remain unfinished.

## Representation and observations

The shared mechanical deformation transports the volumetric material Gaussians
and their compatible deformation gradients. Material covariance describes
physical kernel shape. The distribution over material explanations produces a
separate distribution over predicted motion.

The oracle fixes one H0 point support and the full public rigid hardware. A
single fresh material draw, seed 20261029, comes from the normalized material
marginal of `ConditionalPhysicalPrior`. E is log-uniform; density, Poisson ratio
and body drag are uniform in physical units. The intrinsic log-density includes
the density and drag Jacobians. Tests check normalization and agreement with
the full prior's density ratios.

Observations contain 36 depth rays at each of 0.5 and 1.0 s. Camera poses and
pixel grids depend on public rest geometry, with no deformed-truth mask used
to select pixels. Independent NumPy and Torch intersections check the rest
queries. Missing rays remain censored. The working sensor has independent
0.2 mm range noise and visibility-error probability 0.005. It is an analytic
surface-depth observation model; no rendered RGB encoder is used here.

Three independent 65,536-particle prior importance references also report their
16,384-particle prefixes. Largest-run ESS is at least 12,019; log-evidence spread
is 0.00322. Independent-seed marginal means differ by at most 0.033 posterior
standard deviations. Covariances, quantile shifts and particle-count changes
are reported alongside these figures. ESS alone is not a posterior-accuracy
certificate. Posterior files freeze before opening held motion or material
truth for evaluation; the first seed was fixed for prediction in advance.

Density becomes substantially better constrained. Stiffness remains broad;
Poisson ratio is almost as uncertain as its prior, and drag is weakly informed.
The single true Poisson ratio lies outside its nominal 90% interval, while the
other three material values lie inside theirs. One object cannot establish
coverage or diagnose a coverage failure rate.

## Forward error is measured separately

The material-varying approximation retains 96 nominal vibration modes and three
parameter-independent translation constraint shapes. The plain 96-mode version
missed moving-support deformation: fitting-set moving-support error was
348 µm RMS. Adding the static support response reduced this to 7.5 µm. Both
results are preserved; neither basis used the 64 evaluation cases for selection.

For the fresh oracle's true parameters, full-linear prediction errors are
17.24/10.55 µm for A/B; the enriched modal model has 17.80/11.69 µm errors.
Modal-versus-full errors are 5.25/5.63 µm. In this example the conditioned
prediction's remaining error is larger than these known-parameter errors.
Linearized handle rotations have finite-rigidity defects, which remain in the
reports. This solver is an approximation to fixed-mesh mechanics, not native
Isaac or continuum truth. Its discrepancy is not yet calibrated into a general
unknown-boundary posterior.

## Evidence and reproduction

- [Numerical reference and held predictions](../runs/gaussian-campaign-v01/material-oracle-depth-v01/importance-reference-v01/report.json)
- [Known-parameter controls](../runs/operator-dev/material-oracle-forward-controls-v01/diagnosis.json)
- [Proper-prior comparison](../runs/operator-dev/material-prior-predictive-v01/diagnosis.json)
- [Frozen design](../runs/operator-dev/material-oracle-depth-v01/protocol-design.json)

Raw trajectories, private hypotheses, observations and posterior particle files
stay local. Reconstruct their inputs from the published recipes with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python experiments/reproduce_campaign_material_oracle_inputs.py
```

`--verify` checks existing inputs against the frozen byte hashes without writing.
Run the two `reference-A/B-protocol.json` files through the existing finite
runner with their SHA-256 values from the design, then run:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python experiments/campaign_material_oracle_reference.py \
  --design runs/operator-dev/material-oracle-depth-v01/protocol-design.json \
  --output runs/gaussian-campaign-v01/material-oracle-depth-v01/importance-reference-v01
```

Outputs must be fresh; existing failures and completed evidence are never
overwritten. The separate crossed 96-case Gaussian-network comparison uses its
previously frozen roles and fitting protocol. This oracle does not select its
architecture or checkpoints. GitHub Actions remain disabled.
