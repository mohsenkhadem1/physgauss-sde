# Moving material and localized interactions

The next research direction separates the material that moves from the
interactions that constrain or drive it. Sparse observations should update a
joint distribution over plausible mechanics; each hypothesis produces a
compatible deformation and a prediction of unobserved motion. This is an
extension direction, not a description of a completed general inverse model.

## Representation and current implementation

| Component | Already available | Extension still required |
|---|---|---|
| Volumetric material | Mass/volume-normalized material Gaussians built from tetrahedra; centers and deformation gradients transported by a shared nodal field | Couple inferred spatial materials to a validated forward model and sparse-view inverse |
| Physical kernel shape | Covariance transported as `F Σ₀ Fᵀ`; normalized density preserves represented mass | Resolution checks when deformation varies substantially within one kernel |
| Localized interactions | Reference-surface normalized Gaussian patches, compliant fixture/load responses and separate exact-constraint descriptions | Joint inference of patch locations, extent, type and constitutive parameters |
| Environmental contact | Explicitly separated from material-attached interactions | Validated world-frame unilateral contact and friction; contact is not an attractive Gaussian spring |
| Inference | Numerical mixture over all 55 point-pin node masks and continuous homogeneous materials | General BC types, compliant patches, unknown loads/initial state and image-conditioned hypotheses |
| Acquisitions | Frozen comparison of four additional virtual depth views | Evidence that predictive-risk choices beat fixed/random actions; later controlled mechanical probes |

Material kernel covariance describes physical extent. It is distinct from
posterior uncertainty over positions, material properties or interactions.
Surface appearance can attach to volumetric mechanics; appearance kernels
alone do not supply interior mass or strain energy.

A fixture or load attached to the object uses material coordinates, so it
follows the same material patch. An environmental obstacle is evaluated in
world coordinates, permits separation and requires its own contact law.
Perfect fixation remains an exact constraint; finite fixture stiffness is a
different physical model. Strain gradients must follow the shared motion,
rather than being independent incompatible network outputs.

## Controlled checkpoint and negative calibration result

The independent CPU cohort has completed **96/96 A/B captures for 48 objects**,
with no failed, queued or live physical cells: 5,856 frames in total. Hardware
H0, the point-pin BC type, known force programs and the initial rest state are
supplied. Only support node and homogeneous material are unknown. Each B
program resets from rest; it is not a continuing-state poke. These are finite
120 Hz solver captures, without native-Isaac or continuum equivalence claims.

All 16 calibration episodes completed before the calibration marker and
reserved release froze. The prescribed simultaneous-band multiplier is
**738.5917**, using the 16th calibration order statistic. This very large
inflation is negative evidence about the raw model uncertainty: the declared
joint field needs far wider bands than its latent spread suggests. The score
includes all target mesh coordinates and five fixed prediction states, with a
1 nm denominator floor. It is not a typical-coordinate error estimate and does
not identify a unique failure mechanism. Approximate mechanics, sensor/model
discrepancy and finite-particle concentration must remain distinct possibilities.

All **32 reserved objects completed**, without failures or filtering. Every
policy uses A frames 31–60 and B frames 0–60. Fixed view 3 was chosen on
calibration objects; all four fixed-camera results are retained below rather
than selecting a winner on test objects.

| Policy | Future A pooled RMSE | Reset B pooled RMSE |
|---|---:|---:|
| No additional view | 0.6749 mm | 0.4629 mm |
| Fixed view 0 | 0.3240 mm | 0.2352 mm |
| Fixed view 1 | 0.4270 mm | 0.2877 mm |
| Fixed view 2 | 0.3448 mm | 0.2372 mm |
| Fixed view 3, selected on calibration | 0.4263 mm | 0.2965 mm |
| Random view | 0.3075 mm | 0.2195 mm |
| Predicted-risk active view | 0.3101 mm | 0.2241 mm |

An extra view improves prediction over no extra view, but paired object-level
95% bootstrap intervals for active minus random and active minus
calibration-selected fixed errors include zero. There is no convincing active
selection advantage. The simultaneous bands cover 32/32 objects, exact 95%
binomial interval [0.8911, 1], only with the 738.59× multiplier. Mean active
full band width is **242.8 mm**: coverage is achieved with practically unhelpful
width, not trustworthy raw predictive uncertainty.

Raw nominal 90% material intervals cover E in 27/32 cases, density 24/32,
Poisson ratio 30/32 and drag 29/32. These are descriptive working-posterior
results, not calibrated Bayesian parameter intervals. The true support ranks
first in 13/32 cases, with mean true-node probability 0.352. Numerical variation
also remains: minimum reference importance ESS 89.65, maximum cross-seed support
TV 0.0725, and one extra-view empirical update ESS as low as 2.78/2,048.

A separately frozen **posthoc context** comparison reuses the original C26
observation-independent prior means, with no new forward calls. On these same
32 targets, static-rest A/B pooled RMSE is 2.1411/1.3336 mm and prior-mean error
is 1.6270/1.0389 mm. Both posterior policies improve over these controls;
object-bootstrap descriptive intervals for those differences exclude zero.
This context does not create a new preregistered primary endpoint or establish
active-policy superiority. See the [sealed test](../runs/gaussian-campaign-v01/point-prior-analysis-v01/reserved-evaluation.json)
and [context comparison](../runs/operator-dev/point-prior-context-v01/completed/summary.json).

The earlier [one-object oracle](GAUSSIAN_POINT_SUPPORT_ORACLE.md) improved
prediction but left materials ambiguous and a visible true-parameter forward
error floor. It cannot establish population calibration or policy benefit.

## Elastic excitation development check

On previously seen calibration object 15, the original independent-handle
programs mostly rotate/translate the object. A posthoc pilot applies equal and
opposite known world forces along the public rest-handle separation instead.
Both predeclared amplitudes completed once; the physical solver is unchanged.

| Program | Total motion coordinate RMS | After best rigid fit | Residual / total RMS |
|---|---:|---:|---:|
| Original A | 3,270.61 µm | 5.60 µm | 0.171% |
| Original B | 3,004.26 µm | 5.45 µm | 0.181% |
| Balanced 0.02 N/handle | 2.84 µm | 2.67 µm | 94.0% |
| Balanced 0.2 N/handle | 38.01 µm | 26.72 µm | 70.3% |

The 0.2 N pair increases nonrigid amplitude about 4.8× versus original A while
suppressing bulk motion. Its compatible material-Gaussian field has principal
stretches 0.99298–1.01177, reference-volume-weighted principal-strain RMS 0.0711%
and maximum handle separation change 0.458 mm. Deformation remains modest.
The 26.7 µm nonrigid coordinate RMS is below the stipulated 0.2 mm per-ray
noise scale; these different metrics alone do not establish sensor information
or trustworthy material inference.
Net applied handle force is exactly zero; actual endpoint moments reach
1.03e−7/7.59e−6 N m for the two amplitudes. Rest-central forces do not guarantee
zero torque after deformation. These are applied wrenches, excluding reactions,
drag and internal forces. See the [four-case comparison](../runs/operator-dev/balanced-probe-v01/comparison-v01/summary.json).
This is one-object excitation diagnosis, without posterior, prediction,
calibration or population improvement claims. The sealed reserved evaluation
uses the original unmodified predictor.

Independent supplied-true-parameter controls on calibration objects 0 and 15
still miss examined axial coordinates by 568/906 µm. Full versus modal error
is around 0.9 µm, while near-rigid rotations reach 6.22°/8.11°. The finite-rotation
versus first-order axial-shortening difference explains those examined maxima
to about 1.5–1.6 µm. Retracting saved geometry reduces error but leaves 34–66 µm
overall residuals and roughly 1 µm rigid-patch defects. It is a diagnostic,
not corrected dynamics or validated uncertainty; no reserved retraction was
evaluated. See the [calibration controls](../runs/operator-dev/point-postview-material-audit-v01/completed-findings.json).

A separate manufactured mechanics bridge now adds compliant Gaussian fixtures
to an all-free nodal linear finite-element model. Material-attached weights,
constant world-frame PSD stiffness/damping and prescribed moving targets enter
the implicit solve; finite springs do not replace exact hard constraints.
Four consistency tests and a two-tetrahedron manufactured
case check the assembled force/energy response, CPU time grids and GPU parity.
Errors below 7.4e−16 m in the manufactured displacement check establish that
this small implementation matches its declared model; they do not establish
mesh/time convergence, contact, inference or native equivalence. See the
[bridge report](../runs/operator-dev/compliant-gaussian-manufactured-v01/report.json).

## What should carry into the extension

Retain joint alternative mechanical explanations and estimate conditional
forward discrepancy using independent controls, including supplied-true
parameters. Numerical integration/refinement error must not silently become
material uncertainty. A neural operator remains an approximate conditional
forward model; the completed Gaussian prediction ablation did not outperform
the strong physical baseline.

Unknown forces should be probabilistic when they are unknown. Known probing
commands remain controls. Gaussian force marginalization is exact only for
the stated linear mechanics/linear observation setting; rendered images,
large deformation and changing contact require controlled approximations or
explicit integration. Choose acquisitions for expected hidden-prediction risk,
then evaluate on independent objects with a frozen target and budget.

Evidence: [CPU terminal metadata](../runs/operator-dev/point-prior-calibration-terminal-v01/terminal-verification.json),
[frozen method](../runs/operator-dev/point-prior-analysis-v01/method-protocol.json),
[calibration freeze](../runs/gaussian-campaign-v01/point-prior-analysis-v01/calibration-frozen.json),
[reserved release](../runs/operator-dev/point-prior-analysis-v01/reserved-release.json).
Raw datasets, particles, caches and videos remain local. Pinned inference and
capture sources remained unchanged through the completed reserved run.
