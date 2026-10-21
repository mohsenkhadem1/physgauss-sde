# Gaussian physical hypotheses and active deformation prediction

This implements the final decision in the attached discussion: a conditional
neural trajectory operator replaces a runtime mechanics solver, and an
amortized conditional flow replaces per-video parameter fitting. Isaac/PhysX
generates supervised training trajectories. The initial study uses a known,
metric cuboid and supplied camera calibration, segmentation and static color.
It does **not** establish reconstruction from COLMAP, arbitrary geometry,
real-world material identification or calibrated physical uncertainty.

## Geometry, fields and uncertainty

Let the reference geometry be material coordinates X and surface appearance
Gaussians (mu_i, Sigma_i, color_i, opacity_i). Static surface Gaussians here are
sampled from the known colored mesh rather than learned from real photographs.
Their centers move through barycentric material attachment; local deformation
pushes their spatial covariance forward as F Sigma_i F^T.

These spatial covariance matrices describe the support of rendered splats.
They are distinct from a posterior distribution over physical hypotheses.

A physical hypothesis theta contains positive material coefficients,
support/load centers, force amplitudes and prescribed support motion, plus
boundary and load-family categories. Reference-space Gaussian kernels encode
support and traction fields. A load kernel is normalized with actual surface
area quadrature weights so its integrated traction equals its resultant in
Newtons. Normalizing on an arbitrary neural query set would change the physics.

The present material is spatially uniform: E, rho, nu and linear damping are
unknown; elasticity damping is fixed. The implementation can evaluate support
and load fields at arbitrary points, but this experiment does not yet learn
heterogeneous constitutive fields or arbitrary boundary types. Hard nodal
fixtures in Isaac are only approximated by finite-width neural support kernels.

The common 25D parameter vector has category-inactive padded coordinates. A
canonicalization map removes those coordinates **only for forward prediction**.
It cannot be included inside an invertible change-of-variables density.
Inactive label coordinates receive a declared small measurement tolerance;
their log density is a regularized numerical quantity, not new physical evidence.

A separate structured comparison uses 16 intrinsic cuboid surface-chart
coefficients, a joint 16-way support/load-family category and autoregressive
face categories. Surface coordinates, normal push/pull forces, tangential
shear and equal/opposite perpendicular couples are enforced by its decoder.
Both attached load handles remain part of the mechanical hypothesis, including
when only one is driven. Its density is over chart coordinates and categories,
not a 25-dimensional physical Lebesgue density. Explicit nuisance coefficients
are uniform for inactive controls; finite-face clipping is a pushforward with
boundary mass, outside the invertible density transformation. Geometry-valid
hypotheses do not guarantee accurate parameter inference or positive volumes.

## Direct forward and inverse operators

The learned decoder evaluates

    q_theta(X,t) = X + D_psi(X,t; geometry,theta,controls).

No video embedding enters D_psi. It cannot use image context to bypass the
inferred physical explanation. A t² gate enforces rest position and zero initial
velocity at t=0. There is no global stiffness assembly, equilibrium solve,
FEM invocation or numerical time integration during prediction.

Two variants are compared. A nonlinear trajectory operator directly decodes
displacements. A neural Green operator learns responses nonlinear in material,
support and contact location but linear in applied load and moving-support
amplitudes. Linear superposition is a modeling approximation at this deformation
scale; it does not make finite-deformation elasticity or image formation linear.
Its response to additional calibrated controls requires simulator validation.

An observation encoder produces context c=E_phi(video,cameras,geometry). The
conditional flow maps z~N(0,I) to continuous physical coefficients, with separate
categorical distributions for support and force family. Its Jacobian supplies
the conditional density. The current categorical factorization is an
approximation and can miss support/force category dependence.

The physical map from material and loads to visible images is generally
non-injective. Invertibility of the flow does not repair that ambiguity. In
quasistatic elasticity, scaling stiffness and load together can preserve motion.
Metric geometry, known mass, informative transients and calibrated interventions
can help. A narrow learned posterior is not proof of identification.

For the homogeneous continuum equation
`rho u_tt + rho d u_t = Div(E P_tilde(F,nu)) + f`, simultaneous scaling of
`(E,rho,f)` by a positive constant preserves the equation and fixed prescribed
motions. Additional passive views generally cannot break this scale ambiguity.
Known force magnitudes can. Isaac's two fixed 1g attached handles weakly break
the exact symmetry, so this is a continuum identifiability argument and a
near-symmetry of this dataset, not an exact simulator theorem.

Training currently uses simulation labels and deformation losses; an optional
constitutive/momentum residual exists in the independent reconstruction
baseline. We do not claim a physics-residual-trained proposed operator unless
the corresponding loss and evaluation are actually recorded.

## Observations and comparisons

The primary track consumes RGB, supplied masks, cameras and static geometry.
Explicit silhouette changes are measured relative to the projected known rest
shape. The separately named RGB-D track additionally consumes observed depth.
Neither track may read a deformed mesh, true visibility, physical labels or
withheld pixels during inference. Training supervision and evaluation truth
are separate, explicit interfaces.

Compare posterior predictions with a static object, a matched unconditional
prior, observation ablations, independently implemented RGB reconstruction
models and a geometric RGB-D interpolator. Keep input modalities and known
geometry matched. PIDG-inspired residual fitting must be labeled as such;
it is not an official PIDG reproduction. The exact upstream feasibility audit
and relevant official references are in [GAUSSIAN_BOUNDARY_BASELINES.md](GAUSSIAN_BOUNDARY_BASELINES.md).

Choose checkpoints with validation only. Keep geometry/fixture/load layouts
grouped so matched materials cannot leak across splits. Original test groups
and the fresh 64-layout test set remain locked during development. Report
whole-object and temporal-gap coordinate RMSE, withheld-view image/range
metrics, inversion checks, latency and memory. Foreground metrics accompany
PSNR because background can conceal a poor deformation model.

Posterior validation includes pointwise predictive coverage and interval width,
material/support error, reference-only and unconditional baselines. Coordinates
and frames are correlated: confidence intervals should resample independent
layouts, not individual pixels. Training-label likelihood in mixed physical
units is not by itself a measure of predictive calibration.

## Acquisition rule

For particles theta_s with weights w_s, predict a target Z_s, such as future
deformation, and candidate observations h_a(theta_s). Use a discrepancy-aware
working likelihood

    Y_a | theta_s ~ N(h_a(theta_s), R_a).

R_a includes sensor noise and surrogate/rendering error, estimated without final
evaluation outcomes. Correlated range errors require correlated covariance;
do not treat thousands of pixels as independent precision measurements.

For squared predictive error, current Bayes risk is

    R(w) = sum_s w_s ||Z_s - sum_r w_r Z_r||² / dim(Z).

Choose the action maximizing expected reduction in R divided by acquisition
cost. Fantasize observations from the posterior predictive mixture, condition
the existing weights on each **new** observation, and average the resulting
risk reduction. The implementation reports Monte Carlo error and effective
sample size. A low ESS exposes particle depletion; it is not certainty.

Selection receives predicted observations and public candidate metadata only.
Write the selected action to disk before revealing a measurement. The flow
already conditions on the initial video, so applying its initial-video
likelihood again would double-count evidence. Only the acquired measurement
updates those weights. This approximate particle update inherits any flow and
forward-model errors; evaluate actual prediction improvement rather than just
the model's own reduction in variance.

The saved-state camera pool uses real Isaac renders of an identical state,
including an independent evaluation camera. For causal ordering, initial input
stops at1s, acquisition occurs at1.5s, and future targets lie at2–3s. Physical
camera motion is not simulated: this is a paused-state/repeatable-replay design.
Compare one selected camera against random, fixed-camera and maximum
observation-variance policies at equal one-view budgets.

Calibrated probing is a subsequent repeat experiment with known control
commands and declared contact hardware. Existing rigid load patches affect
inertia and stiffness and must be retained or explicitly modeled. Changing a
contact patch cannot be treated as adding a force to unchanged mechanics.
Known handle positions are extra side information and must be supplied equally
to all policies. Only actual simulated interventions support a claim about
probing; neural counterfactual predictions alone do not.

Probe observation modeling separately scores visibility bits and finite range
returns. Missing returns are omitted range variables. The working model uses
a smoothed Bernoulli visibility likelihood, valid-range covariance submatrices
with their normalization determinants, and a proper broad background mixture.
Fantasies and acquired-data conditioning use the same generative model. This
replaces treating an absent return as a +50mm Gaussian residual. Global noise
parameters must be frozen from reserved TRAIN controls; choosing a noise table
using the true evaluation boundary category would leak the answer.

The frozen controlled-data roles are in
[`gaussian_probe_roles_v01.json`](../configs/gaussian_probe_roles_v01.json):
t0000–t0011 candidate controls fit the teacher; t0012–t0015 candidate controls
calibrate discrepancy. Every diagonal evaluation control and reset replay is
excluded from fitting. Original passive TRAIN videos remain allowed, so the
reserved calibration concerns new interventions, not unseen materials. The
controlled teacher is selected at a predeclared fixed training step. Its
queries retain the original contact hardware/supports and replace the unknown
loads with known commands. They do not advance or integrate a neural state.

The final test design is frozen separately from checkpoint identities in
[`gaussian_final_test_design_v01.json`](../configs/gaussian_final_test_design_v01.json).
All64 fresh TEST layouts are primary; the original12 episodes form a secondary
four-layout comparison. An immutable checkpoint seal is required before the
evaluation runner can open predictive TEST labels. Test acquisition scenarios,
single-action budgets and strong random/fixed/viewpoint baselines are fixed by
that design. Sensor-freshness audits are infrastructure checks, not model fits.

## Evidence status

Development pilots and rejected attempts are preserved under `runs/` and will
be indexed in the final results. Current pilots are not final test results.
The first raw image inverse encoder overfit and failed its motion-information
ablation. Explicit geometry features improve validation conditional likelihood,
but accurate held-out deformation and useful calibrated acquisition remain
separate acceptance criteria. Simulation batching was rejected after it changed
identical-case trajectories; the dataset uses the validated single-process
per-GPU generator. Bird runs remain stopped.
