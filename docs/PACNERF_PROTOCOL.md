# PAC-NeRF elastic transfer: frozen initial protocol

The experiment uses the official `data.zip` and `simulation_data.zip` from
https://github.com/xuan-li/PAC-NeRF (Torus and Bird). The downloader verifies ZIP
CRCs and records SHA256 for every selected member. It accesses only these two
elastic objects, not the unrelated materials or learned third-party reconstructions.

## Scientific scope

This is a controlled transfer experiment conditional on **oracle initial
geometry** (frame-zero particles) and **all eleven initial camera images** for
appearance. It is not monocular geometry reconstruction, not the original
PAC-NeRF evaluation protocol, and not a reproduction of its MPM simulator.
Subsequent particle files are opened only by scoring, after inference. Later
images enter fitting only at the explicitly declared observed camera/time pairs.
Frame-zero geometry is shared by all arms; no material ground-truth values or
future particle trajectories are supplied to optimization, calibration, proposal
construction, stopping, or noise training.

The eleven cameras are fixed. Selecting successive cameras emulates sparse
spatial observations with one image per timestamp, **not a continuously moving
camera capture**. Camera matrices and frame times are the released values.
OpenGL rays are converted to right/down/forward camera coordinates, including
the upstream half-pixel convention. The original images are 800 square; the
first comparison uses 32 square images (LANCZOS resize).

Sixteen cases: two objects, two observation counts, and four inference/model arms:

* 4 images at frames 0,3,10,13; evaluation gap 4..9.
* 6 images at frames 0,2,4,9,11,13; evaluation gap 5..8.
* Camera at selected slot s is `(2*s) % 11`.
* Constant or frozen neural structured internal noise.
* Analytically marginalized or explicitly sampled joint Gaussian applied loads.

The labels "4 views" and "6 views" count the scheduled frame-zero image.
Because all eleven initial images already calibrate appearance and initial pose
is fixed, these regimes supply only **three or five later dynamic images**,
respectively. The frame-zero image contributes a constant likelihood term.

Frames 0..13 are 1/24 second apart. All image pairs after frame zero except
observed pairs are held out for scoring. Initial-camera images are explicitly
excluded from held-out scores. Subsamples from one simulation are not independent
trajectories; these results have only two independent objects/simulations.
The 4/6-image settings differ in camera schedule and gap length as well as
observation count. They are separate sparsity regimes, not a controlled estimate
of the isolated benefit of adding exactly two images.

## Declared transfer mechanics and image model

Deterministic frame-zero FPS selects 128 rendering centers and 32 mechanical
handles. Eight-neighbor central springs define an infinitesimal elastic stiffness;
its twelve lowest modes include six rigid modes and six deformation modes. The
selected basis is frozen in the prepared artifact so eigensolver degeneracy does
not change the model between workers. Gaussian centers follow a fixed RBF blend
of handle displacement. Appearance colors use only the eleven initial images.
Isotropic Gaussian footprints use continuous depth extinction/emission, radius
0.045 m, opacity 0.7, and 24 depth slabs. Known empty-background images are
composited with the object. Initial static-render errors are reported.
This compositing does not model occlusion by the table's depth surface.

The state includes log stiffness scale, log damping scale, initial modal velocity,
and each later frame's modal pose and velocity. Stiffness/damping have standardized
N(0,1) priors with log scale 0.7. The modal elastic system has a matrix-exponential
transition. Gravity is known. A smooth penalty ground reaction with acceleration
`600 * .01 * softplus(-height/.01)` is held at the preceding frame over each
interval. This contact approximation, infinitesimal elasticity, spatial reduction,
fixed Gaussian extent, and coarse image resolution are material limitations.
The inferred stiffness scale is **not a recovered continuum Young's modulus**.

Internal diffusion uses equal/opposite central edge impulses, projected into
the mechanical modes. Their strength is either constant or set by the old
independently calibrated 2–8–1 network using strain and radial speed. Network
weights are frozen. The old constant MLE normalizes neural amplitude; the edge
scale is 0.20 and full-rank acceleration diffusion floor is 0.15, in modal velocity
diffusion units. These are predeclared **transfer assumptions**, not calibration
on PAC-NeRF. The deterministic simulation does not provide stochastic truth.
Four-point Gauss-Legendre quadrature propagates acceleration covariance through
the linear transition. Observation RGB sigma is 0.08, a declared image discrepancy.

Unknown loads have four temporal RBF knots (length 0.14 s), two fixed spatial
profiles (uniform and low-rest-region), and three axes: 24 standardized N(0,1)
coefficients. The acceleration scale is 2 m/s². This restricted load model is
not an arbitrary contact-force field and is not presented as calibrated Newtons.

## Meaning of the marginalization comparison

Both arms have precisely the same force prior, dynamics, renderer, and noise.
The joint arm retains the 24 coefficients. The marginal arm integrates them
analytically with the same Gaussian elimination used in the cube experiments.
Every covariance normalizer is retained. Noise depends on the preceding pose
and velocity, never the next innovation or force coefficient. The state-residual
map is triangular with unit physical-coordinate Jacobian.

Their integrated path posterior is mathematically identical. Different path MAP
estimates are possible; converged posterior means should agree within Monte Carlo
error. Marginalization is an inference-efficiency comparison, not a different
physical explanation. Removing forces altogether would be a separate ablation.

## Inference and evaluation

Observed images and physics alone determine L-BFGS initialization, optimization
in fixed local precision coordinates, damped Newton refinement, and an observed
Hessian proposal metric. Sampling is forbidden unless
the Newton decrement is below 0.02 and observed curvature is positive. A stopped
initial development pilot used only 150 L-BFGS iterations and exposed an
unconverged reference; it is not evidence and motivated this observed-data-only
stationarity gate. Any metric eigenvalue flooring is recorded and affects
proposals only, not the target. Split HMC uses four chains, 500 discarded adaptive
transitions, algorithmic duration U(1,4), target acceptance 0.9, and 2000 retained
draws/chain. Checkpoints at 500 and 1000 retained draws protect progress; they
cannot terminate a final run. Extend the unchanged kernel to 4000 and then 8000
if diagnostics fail. All state coordinates, conditional force means, covariance
normalizers, and log target must have rank/folded split Rhat <1.01,
bulk/tail ESS >=400, and no frozen chains. Failed cases remain published but are
not eligible for scientific rankings. Development pilots use 100+100 and are
always excluded from rankings. **Full nonlinear image gradients and values**
enter the PAC-NeRF Hamiltonian proposals and MH correction. A development-only
transfer of the cube's image Taylor proposal gave an approximately 50% acceptance
ceiling, causing step adaptation toward its minimum. That inefficient surrogate
was removed before the final comparison; its stopped runs are not evidence.

The local-precision optimization was added after direct standardized-coordinate
L-BFGS and small damped Newton steps proved poorly conditioned in development.
It minimizes the same observed-data target. The modal 2x2 matrix exponential is
evaluated by its analytic under/overdamped formula with a smooth zero-argument
series, checked against `torch.matrix_exp`. Batched force elimination and direct
edge-basis contraction implement the same Gaussian algebra without large
intermediate nodal tensors. None of these optimizations changes the model.
`scripts/audit_pacnerf_transition.py` independently checks the linear propagator
and four-point noise/load quadrature against full-matrix and Van Loan
exponentials. Over both initial geometries and five parameter combinations
covering standardized values -3,0,+3, the maximum covariance relative Frobenius
error is 2.85e-7 and maximum load-map relative error is 3.63e-10. This verifies
the linear integration component; it does not validate the reduced material
law, explicit contact approximation, or agreement with the original MPM solver.

Scores: held-out RGB PSNR; particle-coordinate RMSE at 128 fixed frame-zero
indices; componentwise central 90% interval coverage and width in the gap;
runtime and chain diagnostics. Means use every retained draw. Interval quantiles
use at most 2000 evenly selected draws, recorded in each score. Coverage is
descriptive over correlated coordinates, not a repeated-dataset calibration
guarantee. There is no force-error score without independent force truth.
Force coefficient uncertainty is recovered using total covariance E[S]+Cov[m]
in both arms; explicit joint-coefficient sample moments are additionally saved.
An evaluation-only least-squares projection of future truth reports the best
possible particle accuracy in the fixed twelve-mode shape subspace. It never
enters fitting. A separate background-difference-region image PSNR (threshold
0.10) reduces background dominance but is not presented as a ground-truth mask.

Before production outcomes were available, the reporting protocol also added
a paired-sampler agreement screen. The eight equivalent joint/marginal motion
posteriors are compared over 13x12 modal-pose means each using combined
batch-mean Monte Carlo standard errors, with an approximate Gaussian 95%
Bonferroni cutoff across all 1248 coordinates. An overall ranking is withheld
if this screen or the finite-chain checks fail. It is a numerical inference
diagnostic, not a physical-calibration test or proof that no other modes exist.

## Reproduction

```
python scripts/fetch_pacnerf.py --archive data
python scripts/fetch_pacnerf.py --archive simulation_data --pattern '/([0-9]|1[0-3]).ply$'
python scripts/pacnerf_comparison.py --prepare
python scripts/pacnerf_comparison.py --case-index 0
```

Repeat case indices 0..15. `.github/workflows/pacnerf-comparison.yml` prepares
shared inputs once and runs independent cases on GitHub-hosted CPU workers.
It does not assume a GPU is present. Raw author data and full chains remain
downloadable Actions artifacts; the repository stores code, provenance, compact
results, and reports. Dataset access is through the authors' original links.

For an already-running study, collect completed results and full checkpoints:

```
python scripts/collect_pacnerf_actions.py --run 39073611351 --output runs/pacnerf-production --chains
```

This read-only helper uses existing Git authentication, polls every 45 seconds,
and never changes the workflow. Artifact downloads are hash verified. Linux and
Windows raw-root metadata are mapped to inert pure paths when loading our own
prepared artifacts; all tensor bytes and source image hashes remain unchanged.

Before publication, independently replay the raw checkpoints and then publish:

```
python scripts/verify_pacnerf.py --raw runs/pacnerf-production --inputs runs/pacnerf-production-inputs --output runs/pacnerf-production/verification.json
python scripts/audit_pacnerf_contact.py --inputs runs/pacnerf-production-inputs --output runs/pacnerf-production/contact-integration-diagnostic.json
python scripts/publish_pacnerf.py --raw runs/pacnerf-production --inputs runs/pacnerf-production-inputs --output examples/pacnerf-sparse
```

The verifier recomputes position means, interval quantiles, missing-frame scores,
the subspace lower bound, selected scalar chain diagnostics, and cached exact
target values from raw states. All sixteen cases must pass before ranking.
This checks numerical provenance, not physical correctness or global exploration.
Numerical replay and finite-chain convergence are distinct flags: a correctly
replayed unconverged chain is still ineligible for ranking. Interval replay uses
the producer's float32 Torch linspace draw IDs, then independently recomputes
quantiles in NumPy. At 32000 total draws, using float64 NumPy linspace for the
IDs instead would select two different samples; the production scores are unchanged.

After the first production case showed poor transfer, an additional **post-hoc
contact integration diagnostic** compared the zero-noise, zero-unknown-load,
nominal-parameter frame-held contact drift with RK4 at 192 and 384 substeps per
frame for the same reduced ODE. The refined references agree within 3e-13 m,
but whole-sequence particle-coordinate differences are 16.41 mm (Torus) and
17.20 mm (Bird); 4-view-gap differences are 8.18 and 3.44 mm. Thus exact linear
propagation does not make the frame-held contact update accurate. This is not
original-MPM validation or an error decomposition of posterior reconstruction.
The production model remains frozen; the finding limits physical conclusions.

`scripts/audit_pacnerf_noise_transfer.py` optionally compares a fixed subset
of saved posterior states to the documented broad exploratory training component
(strain U(-.25,.45), radial speed U(-1.5,1.5)). It also reports the frozen neural
amplitude divided by the constant control on those states. This is a post-hoc
description, not PAC calibration or a claim about the full training support.
The archived operating-state byte hashes did not reproduce in a local replay,
so that replay is not used as an exact empirical reference. No network weights
or production settings are changed by this diagnostic.
An additional `audit_calibration_replay.py` check reproduces all four archived
cube validation aggregate metrics to at most 2.3e-16 absolute difference, despite
different feature byte hashes. This is consistent with tiny floating-point
variation rather than a material calibration replay failure. It does not
establish PAC calibration or reproduce the original feature bytes exactly.
`audit_pacnerf_posterior_transition.py` additionally checks sixteen stratified
retained states per completed case, independently reassembling nodal diffusion
factors and full-matrix transition/covariance/load integrals. For the first two
completed neural/marginal Torus cases, maximum relative covariance eigenvalue
error is 1.39e-7 and maximum relative load-map error is 6.12e-12. This supports
linear integration accuracy at those probes, not all posterior tails or contact
integration accuracy; the contact audit remains a separate material limitation.
`audit_pacnerf_renderer_reference.py` also renders the supplied true particle
centers and their evaluation-only least-squares modal projection, using the
unchanged initial colors and Gaussian settings. It evaluates the same 138
held-out pairs as the six-view schedule, and never refits appearance. These
diagnostic references expose image-model mismatch; they are neither sparse
reconstructions nor upper bounds on RGB PSNR, since geometrically incorrect
poses can sometimes compensate for rendering errors. No future geometry is
introduced into production fitting by this audit.

For an interrupted production worker, `pacnerf-resume.yml` can download its
saved JSON/reference/chains and continue the **exact original** worker commit,
inputs, adapted step size and RNG state without new warmup. Settings/environment
digests must match or it refuses to run. It is manually dispatched only when
needed. If no chain checkpoint was reached, the saved mode can still avoid
refitting. A restart's reported elapsed seconds cover only that invocation;
retain both run records and use accumulated gradient batches for algorithmic
efficiency comparisons. Resuming does not expand the declared 8000-draw maximum.
The recovery workflow also fetches a SHA256-pinned checkpoint-only wrapper from
commit `0f6bc25984d9da75d7881da056f0dc28708d4ae0`. It imports the original b2d081a
worker, retains its settings, and breaks retained draws into <=50-draw chunks
with atomic file replacement. The first call after fresh warmup saves sixteen
retained draws. Warmup state is saved every 25 transitions and at its boundary,
including dual-averaging accumulators, iteration and RNG. A resumed warmup
finishes the same 500 transitions rather than starting another 500. No additional
warmup is performed after a saved chain exists; diagnostics and stopping remain
at the original budgets. Four nonlinear control tests reproduce the original
sampler's samples, acceptance counts, step size and RNG state exactly, including
interruption during/end of warmup and resumed retained prefixes. Extra
deterministic gradient replays at recovery boundaries are counted.
Each wrapper invocation is recorded separately, without rewriting original
worker provenance. This guards against timeouts; it does not fix poor mixing.

The optional `archive_pacnerf_checkpoints.py --plan-only` verifies local original
artifact ZIPs and writes a reviewable plan without authentication or network
access. Public release uploading has not occurred: automatic approval review
requires explicit human authorization for permanent public prepared-input and
raw-checkpoint assets, beyond the approved ordinary repository pushes. Do not
run publishing mode until that pending authorization arrives. Existing local
ZIPs and extracted checkpoints remain available after Actions artifacts expire.
