# Rotation-aware geometry and joint predictive uncertainty

This stage addresses two diagnosed problems in the previous point-support
experiment: infinitesimal rotations missed near-rigid finite motion, and the
forward model's latent spread omitted substantial mechanical discrepancy.
It uses a **new independent 48-object cohort**. At this checkpoint, new
calibration and reserved outcomes are **pending**; no performance improvement
is claimed from the completed development checks.

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
61 output frames match the previous capture recipe. Capture and calibration
are running; live progress files and episode values are excluded here.

Base inference keeps three independent prior-library seeds and nested
256/1,024 samples per support mask, using exact `p(mask)/Nmask` integration
mass. The fixed first seed supplies 2,048 systematic samples with resampling
seed 2026120700; repeated particles retain their aligned latent conditionals.
One of four prepositioned virtual views at 1.2 s is selected using 32 fantasies.
All policies share A frames 31–60 and B frames 0–60 for action comparisons.

The frozen object-level band score covers no-action and all four potential
view states; calibration, camera selection and explicit reserved release remain
separate. No redraw, retries, category filtering, held-score selection or
post-freeze tuning is allowed. New calibration inflation, reserved prediction,
coverage, bandwidth and active-versus-fixed/random comparisons remain pending.
The [previous 738.59× inflation and negative policy result](GAUSSIAN_MATERIAL_INTERACTION_EXTENSION.md)
are preserved rather than replaced by this development stage.

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
and [old-only development completion](../runs/operator-dev/point-rotation-analysis-v02/development-smoke-v02/terminal.json).
