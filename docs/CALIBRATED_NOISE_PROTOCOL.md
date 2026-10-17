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

### Execution addendum (before held-out test fitting)

The primary 39-motion validation is also executed by
`.github/workflows/validation-study.yml`: four calibration partitions, a job
that verifies all 19 calibration motions and freezes the quantiles, then four
test partitions and a verified aggregate. This changes execution scheduling,
not the model, seeds, observations, budgets or scores. The same frozen network
weights are saved in `examples/calibrated-noise-setup/calibration.json`.

The initial local calibration execution was stopped with 12 completed fits and
no held-out test fitting; its checkpoints are retained as a pilot and are not
mixed into primary validation. Local reference preparation was also stopped.
All primary validation cases use the CI workers' recorded runtimes. Paired
benchmark fits remain local. CPU library versions, worker provenance and
workflow/artifact hashes are recorded. Portable point states reconstruct local
precision matrices without fitting; normalized densities and position/interval
metrics must agree across runtimes within explicit published tolerances.

### Reproducibility corrections and fresh final test split

Cross-runtime recovery of the first CI run failed before publication: its
first reconstructed normalized objective differed by about 25. The spatial RBF
kernel has a triple eigenvalue at the rank-three truncation boundary. Windows
and Linux selected different two-dimensional slices of that eigenspace; row
normalization did not remove the difference in the resulting covariance.
This invalidates treating those worker results as the original fixed physical
model. The initial 19/20 neural whole-gap coverage is a diagnostic only.

Version 0.8.1 freezes the actual cube projection already published in v0.7,
before the v0.8 calibration or tests. The literal constants live in
`src/physgauss/frozen_basis.py` and therefore enter the numerical source hash.
Simulation, calibration and inference all obtain this same basis on every
backend. This is explicitly a fixed rank-three covariance approximation; it
does not recover the full isotropic RBF covariance. A shared prior-rollout
initializer is also serialized for partitioned execution.

An independent curvature check also exposed a zero-extinction derivative error
in rendering. Writing emission as `(1-exp(-tau))*source/clamp(tau)` gives the
wrong emission derivative at exactly zero density. The corrected renderer uses
`source * phi(tau)`, where `phi(tau)=(1-exp(-tau))/tau` and `phi(0)=1`, with a
stable Taylor evaluation near zero. Tests compare first and second derivatives
with the closed-form single-color volume solution, including exactly empty
slabs. This fixes derivatives while retaining the same analytic forward model.

The final complete rerun uses
`configs/calibrated_noise_frozen_basis_study.json`. Its 20 new test seeds are
83001–83020, disjoint from the already examined initial 82001–82020 cases.
The network weights, force/noise settings, 19 calibration seeds, sparse views,
budgets and scoring rules are unchanged. Calibration is recomputed with the
fixed physical projection and frozen before the fresh test jobs. All 36 paired
benchmark fits are rerun with the corrected derivatives. Initial results and
the failed cross-runtime check are retained as diagnostics, not merged into
the final experiment.

For the separate posterior audit, exact observed curvature is assembled from
local joint mechanics Hessians, the force Schur complement, the complete
force-volume logdet Hessian and observed image curvature. Independent dense AD
on a smaller model, full-target Hessian-vector AD, and refined directional
finite differences validate it. The uncompleted full-size brute-force finite
difference attempt is not a result. Four gradient-free Metropolis-corrected
pCN chains, including independent refreshes, replace the initially planned
pCNL kernel to reduce CPU cost; the target, monitored quantities, warmup and
retained budgets remain unchanged. This amendment changes the audit proposal,
never the reconstruction objective or conformal calibration/test procedure.
