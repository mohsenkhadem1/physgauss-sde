# Accurate-reference comparison (v0.6)

The integration and depth-sorting issues found in the previous audit are addressed, and all **20 predeclared fits** have been rerun on independent RK4-96 motion. Both fixes improve reconstruction on the two tested deterministic sequences. Joint and marginalized inference reach similar stationary solutions; alternating inference does not pass the common screen within the same 20-iteration budget. This remains a known-scene synthetic experiment, not a calibrated real-video reconstruction result.

Open [the offline report](../examples/accurate-reference/report.html) for actual images, convergence, force distributions, component controls and view comparisons. [The method/protocol](ACCURATE_PIPELINE.md) explains the new density and renderer. Raw machine-readable results, exact inputs, fitted solver coordinates and portable force distributions are under `examples/accurate-reference/`.

## Numerical accuracy

The centered staggered method uses six substeps per frame. Against RK4-96 it has **0.1297 mm** coordinate RMSE without loading and **0.1298 mm** with the analytic pulse, versus about **3.3 mm** for historical Euler-3. Maximum node/time error is about **0.626 mm**. Twelve substeps reduce these errors to about **0.0323 mm** and **0.156 mm**, consistent with second-order deterministic convergence. The production inference uses six substeps and 1,448 state coordinates; this is not a posterior time-refinement or mesh-convergence study.

The normalized trajectory density includes the drag-dependent triangular Jacobian of the centered kicks. Direct determinant/innovation tests and Schur/profile/marginal identities pass. Gaussian force elimination remains exact conditional on the trajectory.

The renderer integrates absorption and emission through overlapping depth profiles instead of sorting whole primitives. It is permutation invariant, has continuous tie-depth gradients, and retains the old EWA projected footprints. It is a projected volume surrogate, not exact 3D ray tracing. On the six reference views, 128 slabs differ from 1024 by at most **2.155e-5 RGB RMSE**; 512 reference slabs differ by about **1.003e-6**, versus observation noise SD **0.015**. A probe at the historical stalled pulse fit confirms that continuous image changes decrease with step size while hard sorting has rounding-sensitive jumps.

RK4 uses an independent integration routine with the same FEM constitutive law. The preceding audit checked that law against independently differentiated strain energy. These checks do not constitute experimental material validation.

## Matched solver comparison

All three methods use identical new observations, initial state, priors, corrected dynamics and renderer. Position RMSE is over free-node coordinates; the original gap is frames 7-13 (0.35-0.65 s). Fit times include final local covariance construction, on one CPU execution. The final stationarity metric is the SAME marginal Gauss-Newton decrement for all methods; its threshold is 0.05.

| Truth | Method | Gap RMSE mm | Final decrement | Stationary | Force RMSE N | Weight relative ESS | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_load | marginalized | 1.012 | 0.04558 | True | 0.00791 | 0.443 | 35.0 |
| no_load | joint | 1.012 | 0.04558 | True | 0.00791 | 0.443 | 55.6 |
| no_load | alternating | 1.260 | 1.985 | False | 0.01485 | 0.087 | 98.1 |
| external_pulse | marginalized | 1.137 | 0.04719 | True | 0.01139 | 0.123 | 31.3 |
| external_pulse | joint | 1.115 | 0.01392 | True | 0.01130 | 0.133 | 29.0 |
| external_pulse | alternating | 2.538 | 6.51 | False | 0.04218 | 0.022 | 79.0 |

Marginalized and joint inference target the same profiled trajectory optimum here: the force-conditional precision is independent of the unknown scene coordinates. Their small differences reflect finite stopping tolerance and nonlinear steps. Marginalization removes the force optimization block; this two-case experiment does not establish a universal speed or accuracy advantage over joint inference. Alternating inference remains slow under the fixed budget.

All four stationary primary fits pass the predeclared importance-weight screen (relative ESS >=0.1), but the pulse cases only narrowly pass and concentrate substantial weight on individual proposals. These 128 independent proposals are diagnostics of a local Gaussian, not corrected posterior samples or evidence of adequate MCMC exploration. The two alternating fits fail this diagnostic as well as stationarity. No invalid trajectory proposals were silently replaced.

Position intervals are the linear pushforward of the local GN Gaussian. Force moments include both conditional covariance and variation of conditional means over trajectory draws. Reported component coverage uses correlated coordinates from only two sequences; it does not establish calibration. The finite five-knot force basis itself has a dense-grid evaluation-only best-projection RMSE of **0.00287 N** on the pulse (true force component RMS 0.01614 N); this differs from the 21-frame fitted RMSE grid. No oracle coefficients enter fitting.

## Numerical component controls

All rows below use marginalized inference and the SAME continuous-rendered RK4 observations. The hard-renderer controls deliberately mismatch the reference image formation. Dynamics/first-kick discrepancy priors also differ between integrators at finite step size. Do not compare absolute normalized objectives across these different models as evidence.

| Truth | Control | Gap RMSE mm | Stationary | Stop reason |
| --- | --- | --- | --- | --- |
| no_load | corrected | 1.012 | True | step_decrement |
| no_load | legacy | 2.857 | True | step_decrement |
| no_load | integrator_only | 1.258 | True | step_decrement |
| no_load | renderer_only | 1.413 | True | step_decrement |
| external_pulse | corrected | 1.137 | True | step_decrement |
| external_pulse | legacy | 3.649 | False | line_search_failed |
| external_pulse | integrator_only | 1.893 | False | line_search_failed |
| external_pulse | renderer_only | 2.175 | True | step_decrement |

The pulse controls retaining hard sorting fail the line search. The renderer-only control converges but retains the integration bias. Both changes are needed for the best converged reconstruction among these fixed controls. Failed optimizer rows are diagnostic outcomes, not converged estimates.

## Sparse spatial and temporal views

The ten corrected marginalized fits all pass stationarity. The initialization is the same prior-mean, zero-load corrected rollout, with equal native-frame positions on nested solver grids; no truth or images are used to select it. The six original noise residuals are reused, and additional views use fixed seeds. Twelve single-camera times and six two-camera pairs both use twelve images. The +200-degree camera is evaluation only.

| Truth | Schedule | Images | Original-gap RMSE mm | Local 90% gap width mm | Local component coverage |
| --- | --- | --- | --- | --- | --- |
| no_load | six_single | 6 | 1.012 | 9.51 | 100.0% |
| no_load | twelve_single | 12 | 1.156 | 4.61 | 100.0% |
| no_load | six_dual | 12 | 0.821 | 5.11 | 98.8% |
| no_load | twentyone_single | 21 | 0.735 | 3.60 | 98.8% |
| no_load | twentyone_dual | 42 | 0.447 | 1.94 | 97.6% |
| external_pulse | six_single | 6 | 1.137 | 9.50 | 100.0% |
| external_pulse | twelve_single | 12 | 1.227 | 4.61 | 97.6% |
| external_pulse | six_dual | 12 | 1.068 | 5.11 | 97.6% |
| external_pulse | twentyone_single | 21 | 0.786 | 3.60 | 97.6% |
| external_pulse | twentyone_dual | 42 | 0.473 | 1.94 | 97.6% |

Dense observations reduce gap error and local uncertainty in these tests. The twelve-time single-camera fit has slightly worse point error than six times on this noise realization, despite narrower bands. More data need not improve every individual finite-sample point estimate. Dense schedules observe the original missing interval, so their gap score is a common-window comparison, not withheld performance. All-timestamp rows correctly report no withheld-frame metric.

## Provenance, validation and remaining work

- Implementation: `c35b5a16fc163b668f15792362f87a2cd31b9489`.
- Numerical source fingerprint: `fe2687d262d450cd8bf8f7127a3c1fe010c9df14c3e243f36eb80d98f7764702`.
- Fixed plan: `configs/accurate_reference_study.json`; original independent references: `examples/simulation-view-audit/high_accuracy_reference/`.
- All **68** checks and original/new smoke commands passed [implementation CI](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39033558713).
- Publication verifies every saved objective and point trajectory, regenerated RGB/noise hashes, common initialization, force covariance decomposition/positivity, portable force queries, and numerical error gates. `publication.json` records committed source and hashes of published artifacts.
- Local raw checkpoints: `runs/accurate-reference-final`. Full precision factors are excluded from Git; the committed fixed plan regenerates them. Compact solver states and force/trajectory summaries are published for handover.

These tests use one 8-node/6-tetrahedron geometry, 30 Gaussians, 24x24 images, known cameras/appearance/density/supports, fixed discrepancy hyperparameters and a fixed spatial force profile. No arbitrary force localization, unknown geometry, contact or real video is tested. The full-rank nodal discrepancy is distinct from the historical balanced neural SDE. Stochastic-reference validation, nonlinear posterior exploration, repeated-realization calibration, fitted-posterior time refinement and mesh convergence remain open. Historical evidence is preserved rather than overwritten.
