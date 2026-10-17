# Calibrated diffusion and uncertainty validation (v0.8)

This protocol is committed before the full rerun. The older v0.7 measurements
remain unchanged. `configs/calibrated_noise_study.json` fixes all seeds, budgets,
network settings and uncertainty splits. This is a known-scene synthetic study,
not real-video validation.

## Network calibration

The same 33-parameter 2–8–1 tanh network now receives smooth magnitudes of edge
strain and radial speed, scaled by [0.10, 0.50]. Sign symmetry is appropriate for
this declared synthetic diffusion law; it is an assumption to revisit for other
materials. The positive output parameterization is unchanged.

Twelve independently simulated training motions vary initial offset, velocity,
stiffness, damping and pulse presence. Four other motions validate calibration.
Each motion supplies 399 frame/edge states. Equal training weight is assigned
to these operating states and broad random features. At each feature, 128
independent simulated Gaussian innovations supply the empirical second moment.
The loss is normalized heteroscedastic Gaussian NLL. It never regresses directly
on the analytic amplitude. The constant control is fitted to training operating
innovations by Gaussian maximum likelihood; it no longer inherits the old
network's calibration bias. Training is 2,500 fixed Adam steps at learning rate
0.01. Weights freeze before any sparse-view benchmark or interval fitting.

Separate simulator supervision remains a strong assumption. Network-weight,
camera, geometry, material-family and boundary-condition uncertainty are absent.
The training simulator's 48-substep grid supplies feature states only; it is
not used as evaluation ground truth.

## Paired sparse-view comparison

Rerun all 36 v0.7 condition/seed/model combinations, preserving exact observed
RGB and accurate references. Floor, newly calibrated constant, and newly
calibrated neural models all retain exactly marginalized external force and
the same declared full-rank discrepancy floor. The optimizer's iteration cap
increases to 40 for all models, with the same stationarity tolerance. Report
individual cases and the archived v0.7 results; do not select a winning subset.

## Repeated-case interval validation

Use 19 new trajectory seeds for interval calibration, followed by 20 disjoint
test seeds. Each independently draws a stochastic no-pulse/pulse condition with
probability 1/2. Brownian motion and observed RGB noise are independent across
seeds. Reconstruct constant and neural models for all 39 cases (78 fits).
Stochastic reference generation retains the independent 192/384/768-grid
refinement and production 6/12-substep checks from v0.7. All fitting uses only
six observed RGB images at frames [0,3,6,14,17,20].

For each finished fit, compute the maximum absolute standardized position
error over all 84 components (7 gap frames × 4 free nodes × xyz). One complete
trajectory supplies one score. The split-conformal multiplier is order
`ceil((n+1)*(1-alpha))`, with `alpha=0.1`; at n=19 this is the 18th score.
Freeze these two model-specific multipliers before fitting any test case.
Publish raw component coverage, raw whole-gap coverage, conformal whole-gap
coverage, interval widths, and a Wilson interval using 20 independent cases.
Do not treat correlated vertices/frames as 1,680 independent trials.

This constructs prediction sets, not a corrected Bayesian posterior. The
finite-sample marginal coverage property assumes exchangeability of calibration
and new cases and averages over calibration-set randomness. It gives no
conditional guarantee per video, no distribution-shift protection, and no
force/material posterior guarantee. Reference numerical error also remains.
See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

## Nonlinear posterior audit

Keep the full nonlinear marginal-target/local-Gaussian importance screen for
all fits. Separately audit the neural `pulse_and_stochastic__4101` benchmark:
refine its mode to decrement 0.001, form observed Hessians from central gradient
differences in whitened coordinates at epsilon 0.01 and 0.005, and require
relative refinement below 0.001. Reject indefinite curvature explicitly.
Compare Fisher and observed-Hessian importance concentration using 512 draws.

Then run four independently initialized Metropolis-corrected pCNL chains with
1,000 warmup and 2,000 retained transitions each, with 10% independent Gaussian
refresh moves. Warmup-only step adaptation freezes before retention. The target
is the exact nonlinear marginalized-force density; Gaussian curvature only
preconditions proposals. Monitor all 1,448 state coordinates, all 84 gap
components, and full target density. Require rank/folded split Rhat <1.01,
bulk/tail ESS >=400, and no frozen chains. Publish failures rather than silently
declaring posterior uncertainty validated. See
[Vehtari et al.](https://arxiv.org/abs/1903.08008).

Force intervals from local trajectory draws remain labelled approximate and
are scored separately. Conformal position coverage cannot validate them.

## Commands

```sh
physgauss check
physgauss calibrated-noise --quick --output runs/calibrated-smoke
physgauss calibrated-noise --plan configs/calibrated_noise_study.json --paired-source examples/neural-marginal --output runs/calibrated-noise-final
python scripts/audit_calibrated_posterior.py --study runs/calibrated-noise-final/comparison --case pulse_and_stochastic__4101__neural --output runs/calibrated-posterior-audit
```

Stages can run in separate CPU processes with `--stage comparison` and
`--stage validation`. `scripts/prepare_validation_references.py` optionally
prepares disjoint reference partitions using identical source/settings/seeds;
it never fits models or changes the experiment. Full precision matrices/chains
are local checkpoints; compact source, inputs, solver states, diagnostics,
calibration weights and reports are published for handover.
