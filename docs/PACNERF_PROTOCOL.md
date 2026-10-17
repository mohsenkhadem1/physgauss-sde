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

Frames 0..13 are 1/24 second apart. All image pairs after frame zero except
observed pairs are held out for scoring. Initial-camera images are explicitly
excluded from held-out scores. Subsamples from one simulation are not independent
trajectories; these results have only two independent objects/simulations.

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

Observed images and physics alone determine L-BFGS initialization, damped Newton
refinement, and an observed Hessian proposal metric. Sampling is forbidden unless
the Newton decrement is below 0.02 and observed curvature is positive. A stopped
initial development pilot used only 150 L-BFGS iterations and exposed an
unconverged reference; it is not evidence and motivated this observed-data-only
stationarity gate. Any metric eigenvalue flooring is recorded and affects
proposals only, not the target. Split HMC uses four chains, 500 discarded adaptive
transitions, algorithmic duration U(1,4), target acceptance 0.9, and 2000 retained
draws/chain. Extend the unchanged kernel to 4000 and then 8000 if diagnostics fail.
All state coordinates and log target must have rank/folded split Rhat <1.01,
bulk/tail ESS >=400, and no frozen chains. Failed cases remain published but are
not eligible for scientific rankings. Development pilots use 100+100 and are
always excluded from rankings. Full nonlinear rendering enters MH acceptance;
a twelve-coordinate image Taylor expansion accelerates proposals only.

Scores: held-out RGB PSNR; particle-coordinate RMSE at 128 fixed frame-zero
indices; componentwise central 90% interval coverage and width in the gap;
runtime and chain diagnostics. Means use every retained draw. Interval quantiles
use at most 2000 evenly selected draws, recorded in each score. Coverage is
descriptive over correlated coordinates, not a repeated-dataset calibration
guarantee. There is no force-error score without independent force truth.

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
