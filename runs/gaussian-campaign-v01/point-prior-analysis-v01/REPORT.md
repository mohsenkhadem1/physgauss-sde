# Sparse-depth prediction and view selection: sealed 32-object result

Additional depth observations improve forecasting, but the uncertainty-selected
view does not reliably outperform random or fixed views. Simultaneous coverage
requires bands so wide that they are not useful. The calibration diagnosis
locates a concrete forward-model defect rather than attributing every failure
to inverse sampling.

## Frozen experiment

48 independent complete conditional point-pin prior draws, fixed public H0
hardware, known A/B controls and rest state: 16 calibration objects and 32
reserved objects. Both force programs for every object completed once: 96
finite captures, no failures or recaptures. Numerical inference completed all
16+32 objects, retaining every outcome. Raw arrays and clips remain local.

The point-pin type is known; its node and homogeneous material are unknown.
The reference integrates all 55 support masks with exact prior masses and three
independent material libraries. The primary posterior and 2,048 empirical
particles are fixed in advance. Initial observations are 36 noisy depth rays
at each of 0.5 and 1.0 s. One extra virtual depth view at 1.2 s supplies 36 rays.
The active choice freezes before that measurement. Future evaluation uses
only t>1.5 s; there are no later acquired images. B is a new reset-force program,
not a continuing-state poke. Camera travel/backaction and general BC inference
are not tested.

Calibration selects fixed view 3 and freezes a single simultaneous-band scale
before releasing the 32 test objects. All four fixed-view outcomes, the frozen
fixed choice, seeded random policy and active policy are retained. Each object,
including A/B and its potential views, is one statistical unit.

## Prediction

Pooled coordinate RMSE in mm, same target windows for every row:

| Policy | Future A, frames31–60 | Reset B, frames0–60 |
|---|---:|---:|
| Two early depth frames; no extra view | 0.67490 | 0.46295 |
| Fixed view 0 | 0.32401 | 0.23516 |
| Fixed view 1 | 0.42700 | 0.28775 |
| Fixed view 2 | 0.34484 | 0.23719 |
| Fixed view 3, selected on calibration | 0.42629 | 0.29650 |
| Random extra view | 0.30751 | 0.21954 |
| Uncertainty-selected extra view | 0.31009 | 0.22411 |

The extra observation helps. Selection superiority is unestablished: the
10,000-object-bootstrap 95% interval for active minus random mean object RMSE
is −0.0440 to +0.0335 mm for A and −0.0214 to +0.0265 mm for B. Against the
calibration-selected fixed view, intervals also include zero. Fixed view 0 is
best among fixed angles on this test set; it does not replace the prospectively
selected view 3. No model, seed, particle budget or policy was changed from these
outcomes. A separate posthoc rest/prior comparison provides context only: pooled A/B
RMSE is 2.141/1.334 mm for static rest and 1.627/1.039 mm for the preexisting
observation-independent proper-prior mean. This baseline was frozen before
opening sealed target arrays and makes no new inference or physics calls.

## Coverage and physical uncertainty

The prescribed maximum standardized error over mesh coordinates, target times
and five potential prediction states gives a calibration multiplier 738.5917.
It covers 32/32 reserved objects; exact binomial95% interval89.11–100%. However,
mean full bandwidth is **242.8 mm for the active policy**, comparable to the
240 mm block length, and 532.1 mm without an extra view. Coverage bought this way
is not useful local uncertainty or calibrated physical inference. The guarantee
concerns the stipulated sampled target family marginally, not each material,
support, continuum point or robot intervention.

Before an extra view, nominal90% material intervals cover E27/32, density24/32,
Poisson ratio30/32 anddrag 29/32. The true support node ranks first in13/32 cases;
its mean posterior mass is 0.352. The smallest largest-library importance ESS
is 89.65 and maximum independent-seed support TV is 0.0725. Post-view empirical
ESS can be as low as2.78 for one fixed view. Empirical resampled rows are not
independent hypotheses; these diagnostics do not certify convergence or
calibration. The paired before/after material audit uses the same2,048 empirical entries.
Active-view coverage changes E27→28, density24→31, Poisson ratio30→30 and
drag 29→26 out of32. Mean E interval width falls12,789→5,723 Pa; increased
confidence still excludes previously covered truths in two E and three drag
cases. Better support localization (mean true-node mass 0.352→0.676) does not
calibrate all material dimensions. Mean absolute material errors improve
E4,204→1,971 Pa, density69.9→38.5 kg/m³, Poisson ratio0.0298→0.0292 and
 drag0.1325→0.1307 s⁻¹. The active policy has no descriptive material-error
advantage over random or the selected fixed view for E/drag. Active ESS reaches
2.78 in one case (largest empirical weight0.588); this concentration must remain
visible alongside improvement in support-mode accuracy13→25/32.

## Why the worst calibration intervals fail

The worst miss is 0.904 mm in x at a surface point while latent standard
deviation is 1.224 µm. The1 nm denominator floor is not active. Supplied-true
support/material controls retain essentially the same error in the full linear
model. Modal versus full error is below1 µm. Thus material inference and modal
truncation are not the main cause at these coordinates.

Two diagnosed calibration objects rotate by 6.22°/8.11° with little elastic
strain. Finite-rotation shortening predicts the missing x motion within about
1.6 µm. A declared geometry-only retraction of all eight saved true-parameter
A/B/full/modal trajectories reduces full-linear whole-sequence RMSE from
111→41/77→34 µm and 185→66/137→48 µm. It leaves34–66 µm dynamics error and about
1 µm rigid-patch defects. This is a development diagnostic, not corrected
nonlinear dynamics or a model evaluated on the reserved32.

## Implementing the proposed material/interaction split

Compatible volumetric material kernels already move under the shared nodal
field, with physical covariance FΣ₀Fᵀ and conserved mass. A new implicit adapter
now couples separate material-coordinate Gaussian fixture weights and world K/D
laws into stiffness, damping and moving-anchor forces. A two-tetrahedron
manufactured check validates force/work/energy, frame transformations and
kernel transport on CPU/GPU. Its prescribed discrete solution is a consistency
check, not temporal convergence or inference evidence.

A separate development force pilot uses known opposite axial forces along the
public rest-handle separation, on one previously seen calibration object.
At0.2 N, nonrigid RMS grows from 5.6 to 26.7 µm and handle separation changes by
0.458 mm. Net force is exactly zero; moment vanishes at rest but is nonzero
at deformed endpoints, which is measured and retained. This excites more
strain relative to rigid motion but remains modest beside0.2 mm/ray noise.
It does not establish stiffness identification or active probing.

The next candidate applies the rotation correction to each hypothesis before
forming image predictions and uncertainty. It requires a new independent
calibration/evaluation; these32 objects are now development evidence and will
not be relabeled as fresh tests. Compliant-boundary inference, world contact,
RGB encoding and continuing-state force probing remain unfinished. Native
Isaac equivalence and continuum accuracy are not claimed for this FEM cohort.

## Actual-versus-predicted clips

Local videos use original frozen posterior means, a shared orbit/scale and
no deformation exaggeration. Frames before the forecast window are labeled
retrospective. The preselected objects are16 and17; object31 is explicitly the
posthoc worst no-view future-error illustration. All three clips decode to61
frames at1536×576 and20 fps,3.05 s including both endpoint timestamps.

Future no-view→active RMSE: object16 1.100→0.198 mm; object17
0.136→0.148 mm (a worsening); object31 1.423→0.161 mm. The local MP4s and images
are excluded from Git. Their hashes and selection rules are in the
[portable video manifest](../../operator-dev/point-prior-renderer-v01/completed-render-manifest.json).

## Evidence

- [Frozen method](../../operator-dev/point-prior-analysis-v01/method-protocol.json)
- [Calibration freeze](calibration-frozen.json)
- [Reserved release](../../operator-dev/point-prior-analysis-v01/reserved-release.json)
- [Sealed metrics and paired intervals](reserved-evaluation.json)
- [Post-view material audit](../point-postview-material-audit-v01/RESERVED_EVALUATION/summary.json)
- [Posthoc rest/prior context](../../operator-dev/point-prior-context-v01/completed/summary.json)
- [Analysis terminal proof](../../operator-dev/point-prior-analysis-v01/terminal-verification.json)
- [Material and interaction extension](../../../docs/GAUSSIAN_MATERIAL_INTERACTION_EXTENSION.md)

Local arrays, trajectories, particles, renderer outputs and private labels are
excluded from source publication. New checkouts must regenerate local inputs;
source and compact hash-bound reports are published.
