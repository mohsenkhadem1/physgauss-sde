# Rotation-aware geometry and joint predictive uncertainty

This stage addresses two diagnosed problems in the previous point-support
experiment: infinitesimal rotations missed near-rigid finite motion, and the
forward model's latent spread omitted substantial mechanical discrepancy.
It uses a **new independent 48-object cohort**. The 16-object calibration
stage is complete and frozen; the fresh 32-object reserved test is **pending**.
Calibration results do not establish held-out performance or an active-policy
advantage.

## Frozen model

The geometry wrapper retracts each particle's linear-modal motion to account
for finite rotation about its hypothesized point pin. Rays are rendered from
that corrected surface. This changes geometry, not the nonlinear forces,
inertia integration or exact rigid-handle dynamics. Compatible material
Gaussians can be transported by the resulting shared nodal field. Their shape
covariance `F Σ₀ Fᵀ` describes physical extent, not epistemic uncertainty.

A separate rank-16 Gaussian discrepancy couples predicted target motion and
sensor ranges through the **same latent variable**:

```text
target = corrected physical target + mT + BT z
ranges = corrected physical ranges + mO + BO z + sensor noise
z ~ N(0, I)
```

For each material/support particle, base observations condition both its
physical-mixture weight and its latent discrepancy. An extra view uses
`log p(base, view) − log p(base)` and retains the same conditional latent state.
Treating the new view as independent would double-count shared information.
Target discrepancy means/factors are projected to zero at each particle's
candidate point pin. Predictive-risk view selection includes conditional
discrepancy variance and disagreement between physical hypotheses.

The working sensor retains 0.2 mm foreground noise, visibility-error
probability 0.005 and 0.05 m broad background. Visibility is determined by the
physical surface independently of discrepancy: a deliberate approximation to
nonlinear rendering and missingness, not exact image-space physics.

## Completed development work

The discrepancy fit used **only the old 16 calibration objects**, now
development data, with supplied-true parameters and corrected forward
predictions. No new-cohort truth or responses entered fitting. Before fitting,
the retained corrected-forward residuals were:

| Old development program | Coordinate RMS | Maximum coordinate error |
|---|---:|---:|
| A | 27.4 µm | 492.2 µm |
| B | 25.9 µm | 352.0 µm |

The fit completed in 6.235 s. Paired foreground range residual RMS was 3.323 µm;
130/216 channels had no paired foreground training hits and keep zero learned
offsets/loadings. Missing measurements are not invented. These are training
residual diagnostics, without conditional or held-out calibration evidence.

All three corrected observation-independent prior libraries completed in
417.53 s: **168,960 particles**, all 55 support categories retained, no sampled
tet/frame inversions and minimum sampled Jacobian 0.96816. Only old prior
geometry/material coordinates were reused; corrected rays were recomputed.
Old case likelihoods, posterior weights and old rendered rays were excluded.

A first development execution rejected two byte-identical H0 copies because
their paths differed. That rejection is preserved. The loader now verifies
both digests and actual bytes, accepting equal copies at different paths and
rejecting changed hardware; nine synthetic tests pass. The intermediate loader
was not separately published as a reproducible checkpoint. A subsequent full
pipeline smoke on **old object 0 only** completed in 32.3 s. Nineteen root
integration tests passed before execution froze. Neither check is held-out
performance, and neither used new-cohort episode values.

## Fresh evaluation contract and current status

The new proper full prior draws are conditioned only on known point-pin BC0,
with seeds `2026120100+i`, 16 calibration objects and 32 reserved objects.
All 12 continuous coordinates and inactive/discrete nuisance fields are
retained. Public H0, known A/B reset controls, 120 Hz finite integration and
61 output frames match the previous capture recipe. All 16 calibration objects
completed without failure in 527.02 s. Remaining finite captures and the fresh
reserved evaluation are pending at this checkpoint; live progress files and
per-object result values are excluded from publication.

Base inference keeps three independent prior-library seeds and nested
256/1,024 samples per support mask, using exact `p(mask)/Nmask` integration
mass. The fixed first seed supplies 2,048 systematic samples with resampling
seed 2026120700; repeated particles retain their aligned latent conditionals.
One of four prepositioned virtual views at 1.2 s is selected using 32 fantasies.
All policies share A frames 31–60 and B frames 0–60 for action comparisons.

The frozen object-level band score covers no-action and all four potential
view states; calibration, camera selection and explicit reserved release remain
separate. No redraw, retries, category filtering, held-score selection or
post-freeze tuning is allowed. Calibration selected a simultaneous-band scale of
**8.991684563155914** using the maximum of the 16 independent object scores
(rank 16 at the declared 90% level), with the same 1 nm denominator floor.
The fixed camera is **view index 1**, selected solely by calibration A31–60 MSE.
An explicit hash-bound release now permits the 32 reserved objects; their
prediction, coverage, bandwidth and active-versus-fixed/random comparisons
remain pending. This scale is not a direct held-out improvement over the
previous experiment, which used a different cohort and predictive model.
The [previous 738.59× inflation and negative policy result](GAUSSIAN_MATERIAL_INTERACTION_EXTENSION.md)
are preserved rather than replaced by this development stage.

An auxiliary [matched old-model benchmark](../runs/operator-dev/point-old-matched-benchmark-v01/protocol.json)
was frozen before the new reserved release. It will rerun the uncorrected
physical predictor on these same 32 objects, acquired depths, noise seeds,
queries and known controls, comparing the same five potential view states
and common target windows. It has not run at this checkpoint. This is a
paired comparison of complete old and new methods, not an isolated rotation
versus discrepancy ablation or a newly selected old-model policy.


## Calibration-only material and support diagnostic

A separately frozen, post-seal audit uses the **same 2,048 empirical particles**
before and after each view. It retains all 55 support categories and all 16
calibration objects. These are descriptive marginal material intervals, separate
from the calibrated predictive-field bands.

| Parameter | 90% interval coverage, before → active | Mean interval width, before → active | Posterior-mean MAE, before → active |
|---|---:|---:|---:|
| Young's modulus | 13/16 → 14/16 | 14,261 → 8,672 Pa | 3,740 → 2,141 Pa |
| Density | 14/16 → 12/16 | 249.4 → 170.3 kg/m³ | 62.0 → 48.6 kg/m³ |
| Poisson ratio | 16/16 → 16/16 | 0.1264 → 0.1248 | 0.0324 → 0.0321 |
| Body drag | 16/16 → 15/16 | 0.4485 → 0.4387 s⁻¹ | 0.1146 → 0.1073 s⁻¹ |

Narrower intervals can still lose coverage: active acquisition lost previously
covered density truths for two objects and drag for one. Mean true-support
probability rose from 0.319 to 0.626, and support-mode correctness from 6/16 to
12/16, but one active support mode was at least 90% confident and wrong.
The active update's mean empirical row ESS was 452, with a minimum of 22.1;
resampled-row ESS is not an independent-hypothesis count. No model or policy
was tuned using this audit. The frozen fixed camera and random policy remain
required comparators on the fresh reserved test.

A separate video adapter is prepared but has not rendered new-cohort videos.
After the complete reserved aggregate seals, it will select the largest active
improvement, worst regression and worst absolute active forecast error from
sealed A31–60 metrics, breaking ties by object index. It will show true-scale,
common-camera actual/no-view/active panels for **A31–60 future prediction** and
separate **B0–60 known-control rollout from rest**, with no B observations.
The selection is a posthoc illustration, not a representative sample. MP4 and
PNG outputs remain local; compact metadata will record selection and hashes.

Scope remains fixed public H0 and known point-pin type, with unknown node and
homogeneous material. There is no RGB encoder, general BC/contact inference,
continuing-state poke, native-Isaac equivalence or continuum claim. A global
old-data discrepancy model does not establish material/support-conditional
calibration. Raw GP factors, prior archives, private hypotheses and captures
remain local; all live execution source pins are immutable.

Evidence: [method](../runs/operator-dev/point-rotation-analysis-v02/method-protocol.json),
[execution](../runs/operator-dev/point-rotation-analysis-v02/execution-protocol.json),
[old-only fit freeze](../runs/operator-dev/point-rotation-discrepancy-v02/fit-frozen.json),
[completed prior-library summary](../runs/gaussian-campaign-v01/point-rotation-library-v02/summary.json),
[preserved development rejection](../runs/operator-dev/point-rotation-analysis-v02/development-smoke-v01/terminal.json)
and [old-only development completion](../runs/operator-dev/point-rotation-analysis-v02/development-smoke-v02/terminal.json),
[completed calibration freeze](../runs/gaussian-campaign-v01/point-rotation-analysis-v02/calibration-frozen.json),
[explicit reserved release](../runs/operator-dev/point-rotation-analysis-v02/reserved-release.json),
[calibration material audit](../runs/gaussian-campaign-v01/point-rotation-material-audit-v02/CALIBRATION/summary.json)
and [prospective video recipe](../runs/operator-dev/point-rotation-renderer-v02/recipe.json).
