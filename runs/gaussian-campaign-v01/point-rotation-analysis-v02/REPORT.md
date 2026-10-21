# Sealed rotation-aware point-support study

Extra depth views improve prediction on 32 independent reserved objects, but
uncertainty-guided selection does not outperform simple view policies.
The predictive bands cover 31/32 objects with millimetre-scale width; material
intervals still become narrower while losing coverage. A paired old/new
benchmark shows a small forecast change, not a broad physical-inference gain.

## Design and information boundary

The independent cohort contains 48 proper-prior draws, with 16 calibration
objects and 32 reserved objects, conditioned only on fixed public H0 and known
point-pin boundary type. All 96 A/B finite-mechanics captures, 32 primary
analyses and 32 auxiliary analyses completed; no objects were dropped, redrawn
or retried. All source/method/execution pins froze before reserved release.
Each object's predictive states and camera choice froze before evaluation
opened its future response or physical truth.

The predictor uses a rotation-retracted 99-mode linear forward model and a
shared rank-16 Gaussian discrepancy fit only on the previous 16 development
objects. It retains all 55 point masks, three independent prior-library seeds
and nested 256/1,024 draws per mask. The first seed supplies 2,048 resampled
particles. Sensor noise is fixed at 0.2 mm, visibility-error probability 0.005
and broad-background scale 0.05 m. No model was tuned using this test.

Base observations are 72 A-depth rays at 0.5/1 s. A policy obtains one of four
36-ray virtual views at 1.2 s. All policy comparisons use A31–60, **1.55–3 s**,
and B0–60, **0–3 s**. B is an independent reset under a known force program;
there are no B observations and no continuing-state poke.

## Prediction and view policy

All entries below are pooled coordinate RMSE in millimetres, including failures
of prediction and regressions. Fixed view 1 was selected solely on calibration.

| View policy | Future A | Reset B | Mean calibrated full coordinate width |
|---|---:|---:|---:|
| None | 0.7557 | 0.5160 | 5.494 mm |
| Fixed 0 | 0.3900 | 0.2478 | 3.209 mm |
| Fixed 1, calibration choice | 0.3080 | 0.2008 | 3.088 mm |
| Fixed 2 | 0.4274 | 0.2916 | 3.324 mm |
| Fixed 3 | 0.3366 | 0.2239 | 3.070 mm |
| Random | 0.3754 | 0.2459 | 3.053 mm |
| Active | 0.3728 | 0.2429 | 2.921 mm |

Paired differences below average **object RMSE**, a different summary from the
pooled coordinate RMSE above. Negative values favor active. Percentile 95%
intervals use 10,000 complete-object bootstrap replicates, seed 2026120600;
frames and vertices are not independent replicates.

| Active minus comparator | A mean difference [95% interval], mm | B mean difference [95% interval], mm |
|---|---:|---:|
| No extra view | −0.2385 [−0.3933, −0.1052] | −0.1714 [−0.2794, −0.0775] |
| Random | −0.0033 [−0.0425, 0.0358] | 0.0001 [−0.0254, 0.0259] |
| Calibration-selected fixed | 0.0321 [−0.0174, 0.0835] | 0.0264 [−0.0036, 0.0585] |

Extra observations help. Active versus random is inconclusive, and the fixed
camera has lower pooled error. The companion MSE comparison for B favors the
fixed camera: active minus fixed **0.01870 mm²**, 95% interval
**0.00121–0.03955 mm²**. The [aggregate](reserved-evaluation.json) retains all
RMSE and MSE comparisons without selecting the favorable metric.

## Predictive uncertainty, material and support

A single object score takes the maximum standardized coordinate error across
the fixed mesh/time targets and **all five potential states**: no view plus
four possible views. The nominal 90% split-conformal rule with 16 calibration
objects selects rank 16 and scale **8.991684563155914**, with a 1 nm denominator
floor. Reserved joint coverage is **31/32 (96.875%)**, exact binomial 95%
interval **83.783–99.921%**. The one missed object remains in every result.
Mean active full width is **2.921 mm**, not a radius or a standard deviation.
This is population predictive-field coverage over the declared discrete
family; it is not conditional material/support or Bayesian-posterior calibration.

The separate audit compares the same empirical particle rows before/after the
active view. Its unadjusted central 90% material intervals are descriptive.

| Material | Coverage before → active | Mean width before → active | Mean-estimate MAE before → active |
|---|---:|---:|---:|
| Young's modulus | 30/32 → 27/32 | 13,015 → 6,642 Pa | 3,117 → 1,491 Pa |
| Density | 27/32 → 27/32 | 232.2 → 164.1 kg/m³ | 73.2 → 51.0 kg/m³ |
| Poisson ratio | 30/32 → 29/32 | 0.1261 → 0.1222 | 0.0301 → 0.0306 |
| Body drag | 27/32 → 25/32 | 0.4486 → 0.4387 s⁻¹ | 0.1281 → 0.1336 s⁻¹ |

For Young's modulus, four previously covered truths leave the interval and one
enters; drag loses two without gaining any. Narrowing is not proof of improved
physical uncertainty. Mean true-node probability rises **0.498→0.732**, and
support-mode correctness **19→25/32**. Seven active support modes are wrong,
including one at least 90% confident. No support credible-set coverage guarantee
was tested; mode accuracy must not be called coverage.

Active empirical row ESS has mean **464.9**, median **415.2**, minimum **39.3**
out of 2,048. Equal initial resampled weights give row ESS 2,048 by construction,
not 2,048 independent physical hypotheses. Base numerical integration has
minimum importance ESS **62.5** and maximum cross-seed support TV **0.0764**.
These retained diagnostics do not certify posterior convergence.

## Same-object auxiliary old/new comparison

The auxiliary protocol froze before reserved release. It recomputed the old
uncorrected model on the same 32 objects, controls, acquired depths, noise,
2,048-particle budget and **identical view indices**. All old potential states
froze globally before comparison opened primary result values. No old action
selection or old uncertainty-calibration claim is introduced.

| Same-view state | A old → new RMSE, mm | B old → new RMSE, mm |
|---|---:|---:|
| None | 0.7638 → 0.7557 | 0.5136 → 0.5160 |
| Fixed 0 | 0.3990 → 0.3900 | 0.2465 → 0.2478 |
| Fixed 1 | 0.3148 → 0.3080 | 0.2040 → 0.2008 |
| Fixed 2 | 0.4328 → 0.4274 | 0.2922 → 0.2916 |
| Fixed 3 | 0.3534 → 0.3366 | 0.2272 → 0.2239 |
| Primary active indices | 0.3839 → 0.3728 | 0.2433 → 0.2429 |
| Primary random indices | 0.3888 → 0.3754 | 0.2477 → 0.2459 |

For active same-view A, new-minus-old mean-object RMSE is **−0.00867 mm**,
95% interval **−0.01840 to −0.00057**. Its MSE difference is
**−0.00842 mm²**, interval **−0.01866 to 0.00041**, which includes zero.
For B, mean-object RMSE difference is **−0.000925 mm**, interval
**−0.005357 to 0.003335**. Other comparisons and all fixed states remain in the
[matched report](../point-old-matched-benchmark-v01/matched-comparison.json).
These are exploratory paired complete-method comparisons, without a
multiplicity-adjusted blanket superiority claim. They do not isolate rotation
repair from GP discrepancy. Historical cohorts are not used for an improvement
claim.

## Actual versus predicted videos, including failures

The frozen post-seal recipe selects the largest A improvement, worst regression
and worst absolute active error. These are deliberately selected extremes,
not representative samples. Each local MP4 uses shared cameras, scale and colors
at true motion scale, with Independent FEM actual/no-view/active panels.

| Illustration | Object | A no-view → active RMSE | Local clips |
|---|---:|---:|---|
| Largest improvement | 19 | 2.239 → 0.468 mm | [future A](../point-rotation-videos-v02/object-019-A.mp4), [reset B](../point-rotation-videos-v02/object-019-B.mp4) |
| Worst regression | 17 | 0.311 → 0.535 mm | [future A](../point-rotation-videos-v02/object-017-A.mp4), [reset B](../point-rotation-videos-v02/object-017-B.mp4) |
| Worst absolute active forecast | 40 | 1.796 → 0.933 mm | [future A](../point-rotation-videos-v02/object-040-A.mp4), [reset B](../point-rotation-videos-v02/object-040-B.mp4) |

A clips show only 30 future samples; B shows all 61 samples from rest. All six
clips have zero vertices outside fixed plotting bounds. Videos and first-frame
PNGs remain local; [compact metadata](../point-rotation-videos-v02/render-manifest.json)
records their selection and hashes. Earlier conditioned frames are not sold as
forecasting evidence.

## Scope and remaining work

This study uses a fixed H0 mesh/hardware, homogeneous material and known point
support type. Unknown geometry, general BCs, contact, RGB inference, continuing
pokes, robot camera paths, native-Isaac equivalence and continuum accuracy are
not established. Finite mechanics is the declared simulation reference, not
experimental ground truth. The global old-data discrepancy and geometric
rotation correction do not make the forward law nonlinear exact mechanics.

A separate [new nonlinear two-tet reference](../../operator-dev/nonlinear-gaussian-reference-v01/report.json)
passes seven CPU implementation tests: objective finite rotations, independent
energy gradients, tangent/small-force agreement, PSD fixture dissipation/work,
exact constraints, known inactive-load rejection and compatible mass/covariance
transport. Same-load 60/120/240 Hz differences shrink 1.475→0.828 µm. That
solver is not used in this evaluation. Its checks are separate from current
approximate-forward and uncertainty claims.

The completed [durable controller](../../operator-dev/point-rotation-postprocess-v02/status.json)
reports all postprocessing stages successful, with no execution failures. Raw
captures, private parameters, particles and videos remain local. No inference,
calibration rule or policy is changed by this report.

Evidence: [frozen method](../../operator-dev/point-rotation-analysis-v02/method-protocol.json),
[execution](../../operator-dev/point-rotation-analysis-v02/execution-protocol.json),
[calibration freeze](calibration-frozen.json),
[reserved aggregate](reserved-evaluation.json),
[reserved material/support audit](../point-rotation-material-audit-v02/RESERVED_EVALUATION/summary.json),
[matched comparison](../point-old-matched-benchmark-v01/matched-comparison.json)
and [video manifest](../point-rotation-videos-v02/render-manifest.json).
