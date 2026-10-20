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
