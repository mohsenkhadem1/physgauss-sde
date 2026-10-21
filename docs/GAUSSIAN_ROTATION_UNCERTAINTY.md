# Rotation-aware geometry and joint predictive uncertainty

The fresh 32-object reserved study is complete. Extra depth views improve
forecasting, but uncertainty-guided view selection has no demonstrated advantage
over random or calibration-selected fixed views. The joint predictive bands
cover **31/32 objects**, with **2.92 mm mean active full width**. Material and
support inference remain only partly reliable.

Read the [completed scientific report](../runs/gaussian-campaign-v01/point-rotation-analysis-v02/REPORT.md)
for paired confidence intervals, the matched old/new benchmark and failure videos.
All 48 independent objects and their 96 finite-mechanics A/B captures were
retained: 16 calibration objects and 32 reserved objects, with no execution
failures, redraws or post-test model tuning.

## What changed in the model

A geometry wrapper retracts each particle's linear-modal motion to finite
rotation about its hypothesized point pin. It preserves the residual deformation
and exact candidate pin. This changes rendered geometry; it does not solve
corrected nonlinear forces, inertia or exact rigid-handle dynamics.

A rank-16 Gaussian discrepancy couples physical target motion and sensor ranges
through the same latent variable:

```text
target = corrected physical target + mT + BT z
ranges = corrected physical ranges + mO + BO z + sensor noise
z ~ N(0, I)
```

Base depths condition both physical-hypothesis weights and the shared latent
state. A view update uses `log p(base, view) - log p(base)`; it does not treat
correlated observations as independent evidence. Target discrepancy mean and
factor are projected to zero at each candidate pin. View risk includes both
between-hypothesis disagreement and conditional discrepancy variance.

The discrepancy fit used only the **old 16 calibration objects**, now development
data, with supplied-true parameters. It used no fresh-cohort values. Its retained
corrected-forward training residuals were 27.4/25.9 µm coordinate RMS for A/B;
these are training diagnostics, not test accuracy. Of 216 range channels,
130 had no paired foreground training hits and retain zero learned offsets and
loadings. A preserved early development rejection concerned byte-identical H0
copies at different paths; the loader now checks digest and actual bytes.

The working sensor keeps 0.2 mm foreground noise, visibility-error probability
0.005 and a 0.05 m broad background. Physical-surface visibility is independent
of discrepancy, an explicit rendering approximation. Compatible material
Gaussians share the nodal deformation field; `F Σ₀ Fᵀ` is physical shape
covariance, not epistemic uncertainty.

## Frozen evaluation and results

Fresh full-prior draws use seeds `2026120100+i`, conditioned only on known
point-pin BC0 and fixed public H0. Unknowns include the pin node and homogeneous
material; all 12 continuous coordinates and inactive nuisance fields are kept.
Three fixed reference seeds retain all 55 support masks, with nested
256/1,024 material draws per mask and exact `p(mask)/Nmask` weights. The first
seed supplies 2,048 systematic samples. No category or failed outcome is removed.

Inference observes 72 sparse depths from A at 0.5 and 1 s. One of four
prepositioned virtual views supplies 36 additional depths at 1.2 s. Every policy
is scored on **A frames 31–60 (1.55–3 s)** and **B frames 0–60**, where B is a
separate known-control rollout from rest with no B observations. It is not a
continuing-state poke. Predictions and view choices freeze before future
motion or physical truth enters evaluation.

| Policy | Future A pooled coordinate RMSE | Reset B pooled coordinate RMSE |
|---|---:|---:|
| No additional view | 0.756 mm | 0.516 mm |
| Fixed view 1, chosen on calibration | 0.308 mm | 0.201 mm |
| Random view | 0.375 mm | 0.246 mm |
| Uncertainty-guided view | 0.373 mm | 0.243 mm |

The paired mean-object RMSE difference, active minus random, is −0.0033 mm
(95% object-bootstrap interval −0.0425 to 0.0358) for A. Active minus the
calibration-selected fixed camera is +0.0321 mm (−0.0174 to 0.0835). There is
no established active-policy benefit. The report retains every fixed camera and
both RMSE/MSE comparisons; the B MSE comparison favors the fixed camera.

The frozen conformal score takes one maximum per object across the fixed
mesh/time targets and all five potential view states. Calibration chose scale
8.9917 from 16 independent objects. The reserved joint count is **31/32**,
exact binomial 95% interval **83.8–99.9%**, at the declared nominal 90% level.
Mean full coordinate band widths are 5.49 mm without a view and 2.92 mm after
the active view. These population predictive bands do not calibrate material
probabilities or certify material/support-conditional coverage.

On the same 2,048 empirical particles, active acquisition narrows Young's
modulus intervals but coverage falls **30→27/32**; drag falls **27→25/32**.
Young's modulus and density mean estimates improve, while Poisson ratio and
drag remain weakly constrained. Support-mode correctness rises **19→25/32**,
but one active support mode is at least 90% confident and wrong. Mode accuracy
is not support credible-set coverage. Minimum active row ESS is 39.3; repeated
resampled rows do not count as independent hypotheses.

A prerelease-frozen auxiliary benchmark reran the **old uncorrected method on
these same 32 objects**, depths, noise, controls and view indices. Under the
same active-selected views, future A pooled RMSE is 0.384 mm old versus
0.373 mm new. The mean-object difference is −0.0087 mm (95% interval
−0.0184 to −0.0006), while the paired MSE interval includes zero. Reset B is
0.2433 versus 0.2429 mm with inconclusive paired differences. This is a modest
comparison of complete methods, not an isolated rotation-versus-GP ablation;
no claim relies on comparing means from different cohorts.

## Videos and mechanics scope

Six local true-scale, common-camera actual/no-view/active videos show the
largest improvement (object 19), worst regression (17) and worst absolute
active future error (40), selected **after** the sealed aggregate using the
frozen recipe. Each has a future-A clip and separate reset-B clip. They are
illustrations of extremes, not representative sampling. The [video manifest](../runs/gaussian-campaign-v01/point-rotation-videos-v02/render-manifest.json)
records selection, errors and hashes; MP4/PNG files remain local.

A separate [tiny nonlinear Gaussian reference](../runs/operator-dev/nonlinear-gaussian-reference-v01/report.json)
passes objective Neo-Hookean, gradient, Gaussian fixture damping/work, exact
constraint, transport and temporal-refinement implementation checks. It is
**not used by this predictor** and does not validate its approximate forward
law. Finite compliant springs are separate hypotheses from exact hard supports.
No general boundary/contact inference, RGB reconstruction, real camera path
planning, native-Isaac equivalence or continuum convergence is established.

Evidence: [method](../runs/operator-dev/point-rotation-analysis-v02/method-protocol.json),
[execution](../runs/operator-dev/point-rotation-analysis-v02/execution-protocol.json),
[calibration freeze](../runs/gaussian-campaign-v01/point-rotation-analysis-v02/calibration-frozen.json),
[reserved aggregate](../runs/gaussian-campaign-v01/point-rotation-analysis-v02/reserved-evaluation.json),
[material/support audit](../runs/gaussian-campaign-v01/point-rotation-material-audit-v02/RESERVED_EVALUATION/summary.json),
[matched benchmark](../runs/gaussian-campaign-v01/point-old-matched-benchmark-v01/matched-comparison.json)
and [completed durable postprocessing](../runs/operator-dev/point-rotation-postprocess-v02/status.json).
The [previous study](GAUSSIAN_MATERIAL_INTERACTION_EXTENSION.md) remains historical evidence.
