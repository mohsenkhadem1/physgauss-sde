# Force marginalization: completed comparison

Subsequent [simulation/view audit](SIMULATION_VIEW_AUDIT.md): the shared
historical time step has about 3.3 mm error against independent converged RK4.
Measurements below remain valid relative to their discretized synthetic truth,
but do not establish millimetre-level continuous-time physical accuracy.

## Finding

Analytic elimination is numerically correct and removes the separate force
optimization block. It does **not** by itself solve nonlinear inverse rendering.
Under the matched model and tested budget, marginalized inference reaches the
local stationarity screen in 2/4 conditions, joint explicit-force optimization
in 1/4, and alternating updates in 0/4. Reconstruction errors are similar;
there is no consistent accuracy or speed advantage over the joint solver.

Force conditionals are exact for the declared Gaussian model. The overall
trajectory/material posterior is approximate, and the results do not establish
calibration, physical identification or method superiority.

## Protocol and what changed

The [method and equations](FORCE_MARGINALIZATION.md) specify the full model.
All three new solvers share the same observed tensors, fixed force prior,
full-rank velocity discrepancy, nonlinear mechanics/renderer and observation-only
initialization. There are 21 times, 60 solver steps, six observed 24x24 images,
a 0.3–0.7 s gap, 20 maximum outer iterations, 128 predictive draws and 128
separate independent nonlinear-target audit proposals per fit. One geometry
and realization seed 4101 are used; four different truth conditions are not
four independent geometry/seed replicates.

The new discrepancy amplitude is 0.02 (m/s)/sqrt(s), so velocity innovations
have standard deviation 0.02 sqrt(dt) m/s. It is a fixed full-rank nodal
discrepancy, not the old balanced neural internal diffusion. The previous
23-dimensional force-only simulator has a singular trajectory distribution;
the new explicit trajectory model has 728 state coordinates, or 743 with
explicit force coefficients. Thus this comparison distinguishes three matched
solvers, while historical references also differ in their process model.

All inference settings were fixed before the full comparison. A smaller pilot
was used to check execution and numerical identities. No truth or withheld
frames were supplied to fitting, initialization or prior selection.

## Matched point reconstructions

Coordinate RMSE on free nodes at withheld timestamps, in mm. An asterisk marks
a fit passing the marginal Gauss-Newton decrement screen (<0.05). This is a
local stationarity screen, not proof of a global optimum.

| Truth | Alternating | Joint explicit force | Marginalized force |
|---|---:|---:|---:|
| No extra load | 1.122 | 1.057* | 1.057* |
| External pulse | 1.361 | 1.322 | 1.314 |
| Internal stochastic impulses | 3.647 | 3.555 | 3.575 |
| Pulse and stochastic impulses | 3.629 | 3.767 | 3.773* |

In the no-load case, the joint and marginalized state solutions agree
numerically, as expected from their equivalent profiled objectives. Elsewhere,
finite line searches and nonsmooth rendering lead to different endpoints.
The alternating solver's small force-fixed step decrement can coexist with a
large marginal decrement; its own small update does not establish joint
stationarity. The mixed-truth alternating fit uses all 20 iterations; the
other nonstationary endpoints terminate on the declared line-search limit.

## Posterior approximation checks

Relative importance-weight ESS for 128 independent local Gaussian proposals
evaluated against the full nonlinear marginal target. The warning threshold
is 0.1. These are weight-concentration diagnostics, not MCMC ESS, corrected
posterior estimates or calibration tests.

| Truth | Alternating | Joint | Marginalized |
|---|---:|---:|---:|
| No extra load | 0.106 | 0.364 | 0.376 |
| External pulse | 0.050 | 0.063 | 0.024 |
| Internal stochastic impulses | 0.021 | 0.059 | 0.056 |
| Pulse and stochastic impulses | 0.036 | 0.231 | 0.249 |

Only two marginalized fits pass both stationarity and weight-concentration
screens; even those are not calibrated. For the marginalized stochastic and
mixed conditions, nominal 90% coordinate intervals cover only 68.3% and 67.8%
of withheld coordinates respectively. These correlated within-scene coverage
fractions are diagnostics, not population calibration estimates. Correct force
elimination does not compensate for process-model mismatch or local posterior
approximation errors.

There are zero invalid predictive draws among 1,536 requested draws and zero
invalid importance proposals in these runs. The minimum deformation determinant
over predictive frame states is 0.90297; support position error is zero.
Every solver state of each valid draw is also checked by the dynamics residual.

## Force distributions and uncertainty propagation

Marginalized total-load RMSE across time and axes is 0.00620 N (no load),
0.01430 N (pulse), 0.02164 N (stochastic) and 0.02388 N (mixed). Nonzero inferred
smooth loads on pure stochastic truth demonstrate that the model can attribute
unmodelled internal fluctuations to external forcing. These estimates are
conditional and cannot establish the physical cause of motion.

The saved force covariance includes both conditional force covariance and the
covariance of force conditional means over trajectory/material draws. In
standardized force-coefficient trace, the between-trajectory term is about
48–50 times the conditional term for the marginalized fits. Holding mechanics
and motion fixed would therefore omit most of the reported uncertainty. This
is still conditional on known geometry, density, cameras and the fixed load
profile and hyperparameters.

Portable `fields.json` files work with `evaluate_force_posterior`. They preserve
cross-time and cross-axis covariance. Exact observed RGB tensors and shared
initial states are saved in `observations.json`; `point_estimate.json` stores
each full fitted solver-step velocity path and scene parameters. Raw local
Gaussian precisions and paired trajectory/force draws remain in the raw run.
These point/force files alone are not a full joint scene posterior.

## Comparison with the existing simulator

For a common predictive estimand, this table uses **posterior-mean** withheld
coordinate RMSE, rather than the point/MAP RMSE above. All columns use the same
observed images. New Gaussian samples and historical MCMC samples are finite
and provisional; all historical fits fail their complete convergence screen.

| Truth | Old force only | Old SDE only | Old force + SDE | New marginalized |
|---|---:|---:|---:|---:|
| No extra load | 0.989 | 1.747 | 1.980 | 1.088 |
| External pulse | 1.626 | 6.147 | 3.780 | 1.316 |
| Internal stochastic impulses | 3.620 | 2.162 | 2.068 | 3.625 |
| Pulse and stochastic impulses | 3.535 | 5.726 | 2.421 | 3.819 |

The new model improves the selected pulse point prediction relative to the old
force-only reference, but does not improve the stochastic cases relative to
the old combined model. The latter explicitly includes the internal stochastic
mechanism that the current marginal trajectory experiment omits. This table
cannot isolate an effect of marginalization or establish superiority.

## Numerical correctness and timing

- Maximum relative elementwise Schur-complement precision discrepancy:
  1.114e-15 over all 12 fits.
- Maximum normalized joint/profile/marginal density identity discrepancy:
  3.482e-13 in objective units.
- Gaussian-density, gradient, total covariance, SI-unit/time scaling, causal
  Jacobian and old-solver reproduction checks pass in the 57-test suite.
- Sum of measured optimization times over four conditions: alternating 43.64 s,
  joint 20.57 s, marginalized 25.48 s. Shared observation-only initialization
  totals 55.67 s. Final covariance and evaluation costs are additional.

These are CPU measurements from one execution, not controlled timing averages.
Methods can stop at different endpoints; a fast failed fit is not an efficiency
win. Marginalization removes 15 force variables but introduces no guaranteed
speedup over an already coupled joint solver. The matched full study, including
evaluation and shared initialization, used approximately 207 s of recorded work.

## Why fitting still stalls

The post-fit [step scans](../examples/marginal-step-diagnosis/step_scan.csv)
evaluate marginal Gauss-Newton directions at step sizes 1 through 2^-40,
recording actual objective changes and depth-order changes. This is numerical
diagnosis after fitting, not prior or optimizer retuning.

For the pulse endpoint, the first Armijo decrease in this scan occurs only at
2^-34 and coincides with a depth-order change; its magnitude is inconsistent
with the smooth local slope. For stochastic truth, no tested step decreases
the objective sufficiently. At 2^-40, changed depth orders coincide with an
objective increase of about 27.78 despite a smooth predicted change of about
-5.66e-14. Even a near-stationary no-load fit has order-dependent jumps under
some tiny perturbations. These observations directly expose numerical
visibility/depth-tie problems. They do not establish that visibility is the
only source of inference failure.

Next priority: make visibility and derivatives numerically consistent around
depth ties, then repeat the same fixed comparison and validate a nonlinear
trajectory posterior. After this, incorporate the balanced internal SDE into
the trajectory covariance and test discrepancy-prior sensitivity, additional
independent seeds, camera/geometry uncertainty and force identifiability.

## Provenance and reproduction

- Numerical implementation: `e8face70c520855b1a8eaacc4cff2efdc1fee93a`.
- Numerical source SHA-256:
  `9ce4a6f377460fbe0f5378ab3da500da3e6de68ebaaa905c3523e32dd6132a25`.
- `5701c07` corrects diffusion-unit comments/report labels and adds publishing
  and step-diagnosis scripts; numerical executable ASTs are unchanged.
- [Linux CI for the numerical implementation](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39027362092)
  passes all 57 checks and all original/new command smoke paths.
- Raw local run: `runs/marginal-force-study`; the matched completed-boundary
  resume path was exercised without re-running any fit.
- [Self-contained report](../examples/marginal-force-study/report.html),
  [summary](../examples/marginal-force-study/summary.json),
  [CSV](../examples/marginal-force-study/scores.csv) and
  [verified publication provenance](../examples/marginal-force-study/provenance.json)
  are committed with compact inputs, point estimates and force distributions.

Use the commands in the method guide for a fresh execution. To reproduce the
original source fingerprint exactly, check out `e8face7` into a separate checkout
and use its fixed plan. Current source gives a new fingerprint after the units
label correction. Historical source/environment mismatches intentionally stop
resume rather than silently combining experiments.

Publication and diagnosis are reproducible:

```sh
python scripts/publish_marginal_study.py --raw runs/marginal-force-study \
  --output examples/new-marginal-publication --source-ref e8face7
python scripts/diagnose_marginal_steps.py --raw runs/marginal-force-study \
  --output runs/new-marginal-step-diagnosis --source-ref e8face7
```

Publishing verifies the original full source fingerprint, unchanged executable
numerical ASTs, input/initialization identity, checkpoint/summary agreement,
force moments, positive semidefinite cross-time covariance and paired draws.
It retains numerical records unchanged and regenerates report labels. The
figure was inspected visually; the report embeds its image and needs no server.
