# Baselines for Gaussian boundary and material inference

Research and source audit, 20 October 2026. This accompanies the user's attached
discussion: learn a forward deformation operator and a probabilistic inverse
operator from Gaussian geometry and sparse video; predict withheld observations
before choosing additional views or calibrated interactions. It records baseline
feasibility, not completed training or a performance claim.

## What PIDG means

PIDG is **Physics-Informed Deformable Gaussian Splatting: Towards Unified
Constitutive Laws for Time-Evolving Material Field**, AAAI 2026, arXiv
2511.06299. It reconstructs monocular dynamic scenes using a time-dependent
Gaussian deformation field, learned velocity/stress fields, momentum residuals
and camera-compensated optical flow. It is directly relevant to withheld-view
reconstruction. Its published task does not supply a posterior over unknown
fixtures/materials or an acquisition rule for new interactions. A coordinate/time
network fitted to one observed trajectory is not automatically a forward model
of a changed poke. The paper acknowledges costly optimization and incomplete
physical consistency in complex scenes. [Paper](https://arxiv.org/html/2511.06299v1),
[official implementation](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting).

### Exact upstream feasibility audit

Audited commit `b77fd2107a50c875d7219bb4f82388d5e5ea1319`:

- `scene_PIDG/gaussian_model.py` imports
  `motion_utils.time_evolving_material_field.VelocityNet`. That source file is
  absent from the recursive repository tree; only CPython 3.8 bytecode appears
  under `__pycache__`. A clean Python 3.12 checkout cannot resolve that import.
  Another `VelocityNet` exists in `motion_utils/pidg.py`; substituting it is an
  explicit reconstruction patch, not evidence that the missing module matches.
- README asks for Python 3.7.16, while requirements pin Torch 2.5.0+cu121 and
  packages from incompatible Python generations. Use a separate compatible
  environment and record every change. Do not install the full pin list into
  this project's environment.
- Training/rendering needs compiled hash encoding, simple-knn and the external
  flow-aware CUDA rasterizer. This machine's current PATH has neither `nvcc`
  nor `conda`; a driver and CUDA-enabled Torch alone do not provide a compiler.
- `train_pidg.py` selects its best checkpoint using `getTestCameras()` PSNR.
  For our benchmark, give that selection loop validation cameras and preserve
  the final test set separately. Never tune against the published test scores.
- No top-level license was present in this snapshot. Dependencies have their
  own terms. Keep any audit copy separate and do not copy unidentified-license
  implementation into the project.

Sources: [Gaussian model import](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/scene_PIDG/gaussian_model.py),
[requirements](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/requirements.txt),
[training and checkpoint selection](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/train_pidg.py).

An independent CPU check also found an indexing error in the released momentum
residual. The code constructs `grad_v[b,i,j] = d v_i / d x_j` and then uses
`einsum('bi,bij->bj', velocity, grad_v)`. The convective acceleration with that
layout is `einsum('bj,bij->bi', velocity, grad_v)`. For the steady field
`v=(y²,0,0)`, true convective acceleration is zero, while upstream returns
`(0,2y³,0)`. At y=0.2 and 0.3 its y components are 0.016 and 0.054. Calling the
upstream full residual with zero stress and zero viscosity produces loss
0.00052866677, rather than zero. This is a source-level counterexample, not a
measurement of the paper's trained models. [Audited residual source](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/motion_utils/pidg.py).

Exact fetched sources, repository metadata and CPU output are preserved locally
in `runs/baseline-research/`, with `audit.json`. No official PIDG training was
performed in this audit. A small independently written residual model must be
called **PIDG-inspired**, with its deviations listed; it is not an official
PIDG result.

The independently written executable baseline is
`experiments/pidg_style_baseline.py`. It uses a smooth coordinate/time MLP with
Fourier features and exact rest initial conditions; auxiliary velocity and
symmetric Cauchy stress; a Lagrangian momentum residual using first Piola stress;
and an optional compressible neo-Hookean constitutive prior. It fits public
RGB/masks through the shared Gaussian renderer, keeps appearance fixed and saves
the fixed final iteration. It deliberately omits PIDG's hash encoding, flow
rasterizer, optical-flow fitting, densification and the unavailable material
module. Its material prior also differs from the Isaac simulator; this is a
regularized reconstruction baseline, not a constitutive identification result.

```sh
.venv/bin/python -m unittest discover -s tests -p test_pidg_style_baseline.py -v
.venv/bin/python experiments/pidg_style_baseline.py \
  runs/isaac-boundary-orbit-v01/episodes/e0000 \
  --output runs/pidg-style-e0000 --device cuda:3 --steps 500 \
  --query-cameras runs/isaac-boundary-orbit-v01/episodes/e0000/truth/cameras.npz
```

The query camera file provides only requested poses/times, never target pixels
or motion. This baseline's fitter never opens a truth trajectory or image.
Set all three residual weights (`--physics-weight`, `--kinematic-weight`,
`--constitutive-weight`) to zero for its deformation-only ablation. Keep the
same RGB input schedule, architecture, iteration budget and appearance in both.

The declared executable development run uses exactly 200 updates per scene on
e0002, e0005, e0008 and e0015, at 192×160 known-rest ROI with the frozen training
appearance calibration. Physics, kinematic and constitutive weights are each
0.001; the matched deformation ablation sets those three weights to zero.
Both use seed 1701 and the fixed final step, with no hidden images or motion
opened by the fitter. Four physics fits took 110.05, 105.61, 104.15 and 102.97
seconds on GPU3; peak allocated Torch memory stayed below 0.995 GB. The
deformation ablation took 58.69–72.03 seconds per scene on GPU1, with peak
allocated memory below 0.988 GB. GPU3's outer runner stops at a
documented directory-exists handoff after all four successful physics fits,
preventing duplicate queued ablations; `physics/COMPLETE.json` records the four
completed artifacts. This is a resource-queue boundary, not a failed model fit.
Reproduction helpers and logs are under `runs/baseline-research`, with models
under `runs/pidg-style-validation-v01/physics` and
`runs/pidg-style-deformation-validation-v01`. Run with
`TORCH_DISABLE_NATIVE_JIT=1` on this compiler-free machine; the setting is saved
in checkpoint metadata and avoids Torch's optional eager CUDA Triton compiler.

The separate `experiments/rgbd_geometric_baseline.py` is an explicitly
**depth-assisted** per-scene comparison. It fits rigid motion plus 48 smooth
Gaussian RBF controls at observation-time knots using projective depth and
point-to-plane residuals. Correspondences use current predicted normals and
observed depth; unknown simulator material tracks and visibility are excluded.
Depth interpolation is perspective-correct in inverse depth, and observed
normal discontinuities are rejected. Mask support uses a one-pixel quantization
dead zone. Queries use nonuniform Hermite or linear interpolation. Edge,
control, temporal and rest-motion priors regularize invisible directions;
inversions and rejected updates are reported. Photometric fitting is optional
and disabled by default to avoid treating shader discrepancy as motion.

Its four-case development check covers all boundary and force families using
original validation episodes e0002, e0005, e0008 and e0015. The final declared
geometric variant's withheld Gaussian mean 3D distances were 0.162, 0.129, 4.042
and 2.267 mm; static reference errors were 0.105, 0.092, 1.887 and 4.344 mm.
It helps the moving-patch case but is weak for pinned rotation and subpixel
motion. It is preserved as an honest secondary baseline, not evidence of general
recovery. All predictions were finite and uninverted; minimum query tetrahedron
volume ratio was 0.935. These distances differ from per-coordinate RMSE; use
the evaluator's explicitly named common RMSE fields for cross-method tables.
Diagnostics and rejected attempts are under `runs/rgbd-geometric-validation-pilot-*`.

`experiments/ridge_gaussian_inverse.py` supplies a distinct supervised point
baseline: the same public RGB or RGB-D feature sequences regress 25 physical
coefficients plus boundary/force one-hot labels. Feature and label standardizers
fit TRAIN rows only. A dual ridge solve handles the feature dimension; a fixed
lambda grid is selected by validation trajectory RMSE through a frozen neural
forward operator. Runtime inference reads no physical labels. Class scores and
regression residuals are not calibrated posterior probabilities or uncertainty.

On the frozen `gaussian-inverse-fresh-stage01-catalog.json` development catalog
(60 TRAIN, 23 validation episodes), with the same frozen Green-v03 forward
operator, RGB ridge selected lambda 1 and achieved mean dense validation
coordinate RMSE 1.005 mm; RGB-D selected lambda 100 and achieved 0.996 mm.
The static reference scored 1.029 mm. Temporal-gap RMSE was respectively 1.223
and 1.175 mm. Boundary/force classification accuracy was 17.4%/8.7% for RGB and
39.1%/30.4% for RGB-D. The heavy regularization and small predictive improvement
are evidence of limited identified information in this setup, not successful
material/boundary recovery. These validation scores were used for selection;
they are not untouched test estimates. Fits took 51 and 57 seconds on CPU.
All model coefficients, source snapshots, lambda scores, dense predictions and
input/label integrity manifests are in `runs/gaussian-ridge-{rgb,rgbd}-stage01`.

### Matched four-case validation comparison

`experiments/evaluate_gaussian_baseline_predictions.py` evaluates only completed,
hash-checked predictions. It renders withheld orbit frames 25/30/35 and the
actual Isaac independent camera at exactly t=1.5 s, using the same known-rest
ROI, training-only appearance calibration, background and Gaussian renderer for
every model. The table averages four episode scores; coordinate RMSE covers all
61 trajectory times. This is the declared development subset, not an untouched
test estimate. RGB-D methods belong to the secondary input condition.

| Model | Inputs | Coordinate RMSE, mm ↓ | Withheld orbit PSNR, dB ↑ | Same-state new-view PSNR, dB ↑ |
|---|---|---:|---:|---:|
| Static reference | Known rest | 1.439 | 27.862 | 27.800 |
| PIDG-inspired, 200 steps | RGB/masks | 1.840 | 29.522 | 28.357 |
| Deformation-only ablation, 200 steps | RGB/masks | 1.909 | 29.483 | 28.351 |
| Ridge inverse | RGB/masks | 1.390 | 27.982 | 27.900 |
| Neural flow inverse mean | RGB/masks | 1.129 | 28.751 | 27.837 |
| Geometric RBF/ICP | RGB-D/masks | 1.338 | 28.014 | 27.779 |
| Ridge inverse | RGB-D/masks | 1.418 | 27.932 | 27.832 |
| Neural flow inverse mean | RGB-D/masks | 1.199 | 28.140 | 27.719 |
| True motion, privileged renderer diagnostic | Hidden full motion | 0.000 | 29.116 | 28.383 |

The per-scene photometric fitter has **worse mechanics despite better orbit
PSNR**: it even exceeds the true-motion renderer's orbit score by deforming to
compensate for fixed sensor/shader differences. Physics regularization brings
only a small improvement over deformation-only at this budget. Thus higher
photometric score alone is not evidence of material identification or accurate
deformation. The flow improves mean 3D trajectory error on this subset, while
genuine new-view image gains are small for RGB and negative for RGB-D. Preserve
both findings instead of selecting the favorable metric.

All methods' mean trajectories were finite and uninverted. Exact per-case
scores, mask IoU, foreground RGB errors, volume ratios, source/input hashes and
unmodified static/prediction/sensor triptychs are under
`runs/gaussian-baseline-images-validation-v01`; separate privileged diagnostics
are under `runs/gaussian-baseline-images-oracle-validation-v01`. The directory's
`summary.json`, `summary.csv` and four labeled independent-view contact sheets
are produced by `runs/baseline-research/summarize_matched_validation.py`.

```sh
TORCH_DISABLE_NATIVE_JIT=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python experiments/ridge_gaussian_inverse.py \
  --catalog runs/gaussian-inverse-fresh-stage01-catalog.json \
  --operator-checkpoint runs/gaussian-forward-dev-green-v03/best.pt \
  --output runs/ridge-rgbd-new --observation-mode rgbd --device cpu
```

## Other relevant methods

| Method | Relevant capability | Role in this experiment |
|---|---|---|
| GIC, NeurIPS 2024 | Dynamic Gaussian reconstruction, continuum filling, differentiable MPM identification | Strong visual material-identification baseline; iterative solver comparator |
| PhysTwin, ICCV 2025 | RGB-D perception, spring-mass inverse modeling, Gaussian appearance, unseen interactions | Strong interaction baseline after adapting the input/controls interface |
| PhysGaussian, CVPR 2024 | Gaussian reconstruction coupled to forward MPM with specified physics | Supplied-parameter forward simulator or oracle; not inverse identification by itself |
| PhysDreamer, ECCV 2024 | Distills video-generation priors into material fields for plausible interaction | Generative-prior comparison, not measured material ground truth |
| Frozen DINOv2 ViT-S/14 | Generic image/patch descriptors, 21M parameters | Optional perception feature ablation; no force, material or uncertainty oracle |

GIC compares reconstructed surfaces and rendered masks with a differentiable MPM
trajectory. Its published inputs are multi-view video and its objective estimates
physical parameters; it does not solve general unknown fixture identification.
The official environment uses Python 3.9, Torch 2.4/CUDA 12.1, Taichi 1.2,
PyTorch3D and compiled Gaussian dependencies. The root license is MIT; dependency
licenses still apply. Audited commit:
`b523851ed4343109ff67b0ea123e0a154af2a40c`.
[Paper](https://arxiv.org/html/2406.14927v1),
[source and environment](https://github.com/Jukgei/gic).

PhysTwin is closer to the intervention task: it optimizes spring-mass physical
parameters and geometry from sparse RGB-D views, then simulates new interactions.
Its current Linux install targets Python 3.10, Torch 2.4/CUDA 12.1 and Warp, with
additional perception/rendering dependencies. Known synthetic geometry and masks
let us bypass the TRELLIS, segmentation and camera-hardware preprocessing, but
this bypass must be disclosed. Root license MIT. Audited commit:
`81c718790a37e5e0102eb77af2c6edd34a9db25f`.
[Paper](https://arxiv.org/html/2503.17973v1),
[implementation and installation](https://github.com/Jianghanxiao/PhysTwin).

PhysGaussian supplies a forward MPM model with material and boundary parameters
specified by the user. It is useful as a forward reference, not a learned inverse
posterior. Its snapshot uses Python 3.9, Torch 2.0 and Warp 0.10.1. PhysDreamer
instead obtains plausible material behavior from a video diffusion prior; its
repository prioritizes pretrained inference and also contains uncleaned training
code. Neither root snapshot declares a license. Audit commits respectively
`8339ed6aa2cd5d50e1001a254a3d95aea678a956` and
`95beb71f9f1b037b05d7c364d1943dde10b0efbd`.
[PhysGaussian source](https://github.com/XPandora/PhysGaussian),
[PhysDreamer paper/project](https://physdreamer.github.io/),
[PhysDreamer source](https://github.com/a1600012888/PhysDreamer).

A frozen small DINO feature extractor is feasible as an optional observation
encoder. Compare it with the identical model using plain RGB/geometry features;
do not assume semantic features resolve repeated texture, hidden motion or
material ambiguity. [Official DINOv2 models and terms](https://github.com/facebookresearch/dinov2).

## Fair executable comparison

The current orbit dataset supplies a known metric volumetric rest mesh and
surface-attached Gaussian scaffold. It is not a COLMAP reconstruction or a
trained 3DGS asset. COLMAP estimates camera poses and sparse/dense geometry;
Gaussian fitting is a subsequent stage. Benchmark controlled mechanics first;
report reconstruction noise as a separate ablation. [COLMAP documentation](https://colmap.github.io/tutorial.html).

Use the same allowed observations for every non-oracle method. Start with
RGB-D, masks and known cameras if that is what perception actually consumes;
label this explicitly. Geometry tracks extracted from RGB-D are observations.
Tracks projected from saved simulator motion are privileged correspondence
supervision and require a separate oracle-track condition. Do not compare an
RGB-only literature baseline against a proposed model supplied hidden 3D states.

The minimum useful measured comparison is:

1. Rest geometry, rigid motion and linear/smooth temporal interpolation.
2. Independently fitted geometric deformation with the same observations.
3. Deterministic neural deformation/physics model with the same training data.
4. Boundary/material posterior predictive mixture.
5. Posterior plug-in/MAP ablation, separating the benefit of uncertainty.
6. Correct-boundary and fully known-parameter oracles, clearly privileged.

Keep group splits frozen: matched stiffness triplets remain together. Hide test
recipes, boundary assignments, material values, full trajectories, dense frames
and visibility truth from inference. Training may use synthetic parameters and
motion only when that supervision is declared and shared with corresponding
supervised baselines. The 48 episodes contain only 16 independent layouts and
one shape; they cannot establish broad shape or real-world generalization.

Report temporal-gap completion separately from genuine new-view prediction.
The existing one-camera orbit records only one view at each time; querying
another point on that same orbit at another time mixes viewpoint and temporal
errors. Render additional withheld camera poses at the *same saved state* for a
novel-view test. Use no test-view pixels to train appearance or choose checkpoints.

Evaluate hidden/visible Gaussian-center error, withheld foreground RGB/mask/depth
errors, inversion/boundary violations, and uncertainty using proper scores and
interval coverage across episodes. Report perception failures and renderer
discrepancy separately. A deformation model should not win by changing texture.

### Shared observation model and its development diagnostic

`experiments/gaussian_observation.py` supplies the shared differentiable renderer.
Its optional 192×160 ROI projects the **known rest geometry** and adds a fixed
30 native pixel margin, then adjusts each camera's intrinsics for its crop.
It does not use the dynamic truth mask's bounding box. Gaussian centers are
surface-attached; spatial covariance is transported by the incident tetrahedron's
deformation gradient. Appearance is fixed across times and material classes.

One global affine Gaussian-color correction was fitted from 117,075 interior
pixels in the 24 training episodes' frame-zero images. It is frozen for all
validation/test use. On the development diagnostic's 252 train/validation query
frames, static reference predictions scored 28.216 dB PSNR, 0.94283 mask IoU and
0.05993 foreground RGB RMSE; the privileged true-motion renderer scored
28.812 dB, 0.95083 and 0.05541. These measure renderer discrepancy and available
motion signal, not learned-model performance or a universal attainable ceiling.
The diagnostic is in `runs/gaussian-observation-roi`; reproduce with
`experiments/diagnose_gaussian_observation.py --fit-appearance` and its CLI
dataset/output arguments. The test split was not used for this diagnostic.

### Sensor freshness and exact-state repair

The legacy dataset validator checks depth rays at five of the 61 frames. An
additional all-frame audit now checks 32 rays from the projected known-rest
interior and 32 from the saved mask interior at every frame, against independent
triangle intersections with the saved nodal state and camera. Geometry-predicted
background is checked as background when deformation leaves the rest projection.
The unchanged maximum depth-error bound is 0.2 mm. This audit is simulator QA,
including withheld pixels only for sensor verification; it supplies no hidden
states or target pixels to fitting or action selection.

The initial snapshot audited 154 completed episodes: 106 fresh and all 48
original episodes, for 601,216 rays. After repairing fresh t0053, 149 passed and
five isolated stale frames were detected: fresh training t0054/frame53,
t0057/frame31, t0060/frame19, t0063/frame9, and original validation e0040/frame57.
Their maximum ray errors were 4.81–8.55 mm. All five are unobserved frames, so
their sparse model inputs are unaffected. Full reports and exact sensor hashes
are in `runs/isaac-all-frame-sensor-audit-v01`; a CPU-only resumable watcher in
`runs/baseline-research/watch_saved_sensor_audits.py` follows later completions
under `runs/isaac-all-frame-sensor-audit-v02`.

`experiments/repair_isaac_dataset_sensors.py` preserves the entire rejected
episode, original status/log and all checksums under each dataset's
`rejected-attempts/stale-sensor/`. It creates an independent working copy and
recaptures only failed frames from the **exact saved q and pose**, using one
paused static Isaac mesh, the original colors and dome light, and no dynamics
steps. `rep.orchestrator.step(delta_time=0, wait_for_render=True)` provides an
explicit annotation completion barrier; copied RGB/depth readbacks must pass
current-state rays before they can replace pixels. Matching sparse copies are
refreshed when necessary. Physics/reference/camera/recipe bytes and original
worker source attribution remain unchanged; replacement pixels have their own
source snapshot, checksums and repair lineage. Promotion requires all-frame
auditing and the unchanged existing full validator. No threshold is relaxed.

t0053 had 36 stale frames and was repaired successfully in 28.55 seconds,
including startup and independent validation. Every recapture passed its first
attempt, with depth errors around 0.2 µm. Its original remains byte-identical.
The five additional one-frame repairs also passed in one static Isaac batch,
about 36 seconds including startup and full CPU validation. All passed their
first readback, with maximum depth error 2.02×10⁻⁷ m. Their originals and exact
file checksums are preserved; no physics was resampled. The original
e0040 failure predates current Torch sharing, so sharing is not established as
the sole cause of asynchronous sensor readback. The four-case matched image
comparison above does not contain e0040 or these fresh training episodes.

## Active observation and probing

New-view selection needs a likelihood for possible observations under each
boundary/material hypothesis and an expected information or predictive-risk
utility. Compare that choice with random view, largest baseline and purely
geometric visibility acquisition at equal cost. Choose an action before exposing
its observation. Reveal the selected measurement once, update the posterior,
then score on a separate untouched prediction target.

For poking, generate a **new simulator rollout** from the declared state under
the chosen force or displacement command, keeping the hidden material/support
fixed. Re-rendering an existing trajectory or appending a prescribed displacement
without observing its force response does not establish active mechanical
identification. Compare random poke and fixed poke, and report both information
gain and later deformation error. A coordinate/time reconstruction baseline has
no defined new-poke prediction unless it is extended to condition on controls.

An actual same-state new-view pool now exists for all 12 original validation
episodes in `runs/isaac-acquisition-validation-v01`. At exactly t=1.5 s, its four
candidate cameras use azimuths −135°, −45°, 45° and 135°, elevation 25° and radius
0.92 m; the independent evaluation camera uses 75°, 43° and 0.97 m. Poses depend
only on the known rest center. A sixth source-camera replay is a sensor audit.
The generator directly renders the saved colored mesh through Isaac with the
same dome-light settings; it performs no dynamics steps and preserves original
episodes. Thus all five task views observe the same physical state.

All 72 RGB/depth/mask captures passed 64 independent mesh-ray checks each:
maximum depth disagreement 3.36×10⁻⁷ m, no sensor retries. All source-pose replays
had mask IoU 1.0; their maximum RGB RMSE was 0.00503, reflecting a small RTX
accumulation difference. This pool establishes sensor/state consistency, not an
active-policy improvement result. `load_public_view_queries()` opens only camera
metadata; `selected_candidate_paths()` exposes a selected candidate and rejects
evaluation views. Hidden images and state stay under the separate `truth/` tree.

```sh
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaac/python -u \
  experiments/isaac_acquisition_views.py \
  --dataset runs/isaac-boundary-orbit-v01 \
  --output runs/isaac-acquisition-validation-new --split validation --gpu 3
.venv/bin/python -m unittest discover -s tests -p test_isaac_acquisition_views.py -v
```

The attached no-FEM proposal is compatible with a conditional neural trajectory
operator and a conditional flow over low-dimensional material, support and load
coefficients. The flow is invertible; the physical video map is generally not.
Keep the forward decoder conditioned on physical coefficients and controls,
rather than unconstrained video features that bypass them. Calibrated known
forces or informative dynamics with known mass are needed to distinguish
material stiffness from arbitrary unknown force scale. Foundation features and
a flow cannot remove that structural ambiguity.
