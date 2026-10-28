# Official PhysTwin: first physics baseline

This adapter runs the MIT-licensed [official PhysTwin](https://github.com/Jianghanxiao/PhysTwin)
at commit `81c718790a37e5e0102eb77af2c6edd34a9db25f`. The checkout, inputs,
weights, environments, and videos live under ignored `runs/` directories.

## Setup

From the project root, with `uv`, Git, and an NVIDIA driver available:

```bash
bash scripts/phystwin/setup.sh single_lift_sloth
```

Python 3.10, PyTorch 2.4/CUDA 12.1 and Warp 1.6.2 are isolated from our existing
environments. Only selected members of the official Hugging Face ZIP archives
are downloaded. ZIP CRCs and SHA-256 manifests identify the retained inputs.
The exact installed package versions are retained with the experiment report.
Pickled inputs/checkpoints are trusted upstream artifacts, not arbitrary uploads.

## Run

```bash
# Reproduce the authors' checkpoint and compare to their saved trajectory.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  runs/phystwin-env/bin/python scripts/phystwin/run_physics.py \
  --mode reference --output runs/phystwin-pilot-v01/reference

# Fit dense physics parameters from the authors' first-stage initialization.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  runs/phystwin-env/bin/python scripts/phystwin/run_physics.py \
  --mode train --iterations 200 --output runs/phystwin-pilot-v01/retrained

# Independently fit first-stage topology and physical parameters from defaults.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  runs/phystwin-env/bin/python scripts/phystwin/run_cma.py \
  --iterations 20 --output runs/phystwin-pilot-v01/cma

# To train stage two from a new first-stage result, add:
# --initialization runs/phystwin-pilot-v01/cma/optimal_params.pkl
```

Use a fresh output directory for each run. Change `--case` in both download and
run commands for other upstream examples. No W&B account is required; external
experiment logging is disabled. GPU 3 is reserved for the desktop.

## What is and is not changed

`headless-imports.patch` guards the optional interactive/Gaussian-renderer imports.
The runner replaces only `visualize_sim` with the same forward rollout and a CPU
point-cloud video. It uses the upstream topology constructor, Warp kernels,
losses, training loop, optimizer, time step, and training-based checkpoint choice.
The first-stage runner suppresses optional OpenGL previews.

The videos show observed visible surface points versus predicted surface points,
with fixed shared projection/scale and red prescribed control points. They are
not photorealistic Gaussian renders. Training/future frames are explicitly marked.

These runs start from author-processed RGB-D observations, tracked points and
completed geometry. They do not reproduce raw RGB-D preprocessing, TRELLIS,
tracking networks, Gaussian appearance training, or our own object dataset.
Known future controller motion is supplied; this is action-conditioned forecasting.
Offline preprocessing is inherited from upstream and has not been audited for
strictly causal tracking. This is not yet an unseen-interaction transfer study.

`summary.json` records frame splits, hardware/software, checkpoint/source identity,
and train/test metrics. Manual tracking uses first-frame nearest-particle
correspondence and mean Euclidean distance. Surface error matches the official
one-direction L1 Chamfer convention, implemented with SciPy KD-trees. Missing
manual annotations are excluded rather than counted as zero error. The provided
examples have no fully missing manual-track frames in the evaluated splits.

The stationary first-frame baseline is only a basic sanity check. It is not a
strong competing model. Test metrics are diagnostics and never select a model.
Warp warns that controller interpolation has no backward kernel; controllers are
prescribed inputs here, not fitted parameters. Physics-parameter gradient
correctness is checked in one direction with `check_gradient.py`, not proved
for the full system. The CUDA graph captures the first-frame visibility-count
normalizers; eagerly calling `calculate_loss` later uses updated counts. An
initial audit compared these different objectives and showed a 17% discrepancy.
The corrected audit uses the identical captured graph for both autodiff and
finite differences (about 0.3–1.5% discrepancy). We preserve the upstream
normalization behavior for reproduction.

`audit_results.py` additionally compares persistence and constant velocity from
the final fitted prefix states, and removes best-fit rigid motion from the
manual tracks. Those temporal baselines are diagnostics sharing the fitted
prefix, not separately trained competing methods. Their inputs stop at the
training cutoff. Sparse nonrigid residuals are not dense strain estimates.

Metrics tests: `.venv/bin/python -m pytest -q tests/test_phystwin_metrics.py`.

## Controlled attachment transfer study

See [the controlled report](../../reports/phystwin-transfer-v01/REPORT.md) for the frozen design, reproduction commands, held-interaction results and negative uncertainty/probing results. `controlled-physics.patch` adds an opt-in gravity scale (default 1 preserves upstream behavior). `controlled_reference.py` generates independent nonlinear FEM references; `run_controlled.py` adapts known geometry and masses to the official dense fitter; `controlled_probing.py` runs a separate finite FEM posterior, not PhysTwin uncertainty. Raw trajectories and videos stay local.

## Continuous uncertainty reference

[The v02 uncertainty report](../../reports/phystwin-uncertainty-v02/REPORT.md) replaces the finite physical prior with continuous parameter quadrature and an independently calibrated numerical-discrepancy model. It tests 24 held-out material/support pairs after 12 development and 20 calibration pairs. This is an FEM posterior reference, not learned PhysTwin uncertainty. `uncertainty_reference.py`, `uncertainty_posterior.py`, `freeze_uncertainty.py` and `summarize_uncertainty.py` implement the resumable study; the report contains reproduction commands and limits.

## Active probing across unknown fixture families

[The v03 support-policy study](../../reports/phystwin-active-support-v03/REPORT.md) adds left/right, full/half-face compliant attachments and competing reset pushes. It freezes calibration-selected fixed and uncertainty policies before 24 test objects. Adaptive probing beats fixed/random, but does not beat the simpler inferred-support heuristic. The report also exposes substantial mesh sensitivity. `active_support_reference.py`, `active_support_policy.py`, `audit_active_support.py`, `refine_active_support.py` and `summarize_active_support.py` reproduce the reference study; it is not a learned PhysTwin policy.

## Difficult-fixture reference qualification (v04)

`consistent_fixture.py` integrates Robin boundary energy with a consistent
surface mass matrix and offers quadratic tetrahedra, grading near the fixture
edge, FP64 CUDA assembly, and CPU sparse Newton solves. Material and loading
remain those of v03; this is reference qualification before further inference.
`qualify_fixture.py` writes local fields; `summarize_fixture.py` applies the frozen
displacement, strain, traction, energy, equilibrium and quadrature gates.

```bash
.venv/bin/python -m pip install -r scripts/phystwin/requirements-fixture.txt
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/phystwin/qualify_fixture.py --degree 2 \
  --levels 8 12 16 20 --action initial --device cuda:0 --grading --linear-solver pardiso
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/phystwin/qualify_fixture.py --degree 2 \
  --levels 20 --order 4 --action initial --device cuda:0 --grading --linear-solver pardiso
.venv/bin/python scripts/phystwin/summarize_fixture.py
```

Repeat for `push_left`, `push_right`, and `held_oblique`. Fine meshes need substantial
RAM; limit concurrent jobs to available memory. GPU 3 remains reserved.
See `reports/phystwin-fixture-convergence-v04/REPORT.md` for measured scope and limits.

## Qualified reference, uncertainty and probing campaign (v05)

`qualified_reference.py` resumes a bank or per-object spatial/quadrature
qualification. `qualified_policy.py` selects a forward surrogate using development
objects, converges continuous posterior moments and physical quantiles, calibrates
prediction bands, seals the analysis, then compares fixed/random/adaptive/fixture-side
policies. Physical-parameter coverage and interval width are audited separately.

```bash
# One worker per physical GPU 0–2; use worker indices 0,1,2 with --workers 3.
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/phystwin/qualified_reference.py \
  --split development --worker 0 --workers 3
# Other splits: bank (optional --bank-mesh 8 or 12), calibration, test.
# Selection requires all development objects to qualify; missing finer banks
# are requested explicitly rather than silently accepting a coarse surrogate.
.venv/bin/python scripts/phystwin/qualified_policy.py --stage select
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/phystwin/qualified_policy.py --stage development
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/phystwin/qualified_policy.py --stage calibrate
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/phystwin/qualified_policy.py --stage test
```

The v04 thresholds are retained. An analytic zero-strain translation check handles
same-face pushes on full supports; relative roundoff errors are not meaningful
there. Refinement and surrogate-selection addenda are preserved in the report.
GPU multigrid preconditions the current tangent without changing the equilibrium
problem; an independent CPU residual check remains mandatory. Quadrature-gradient
and tangent-contraction streaming limit GPU working memory. Raw checkpoints stay
local. A failed object cannot silently disappear from calibration or testing.

## Multi-support sequential probing and RGB-D / PhysTwin bridge (v06)

[The v06 report](../../reports/phystwin-sequential-support-v06/REPORT.md) contains
36 displacement-qualified references, a three-parameter continuous posterior,
six force actions, two sequential probes, a rendered RGB-D fiducial observation
model, and an isolated official-PhysTwin known-force adapter. It reports benefits
at each observation budget, numerical sensitivity, coverage, and negative results
without reducing the whole study to one pass/fail label. v05 inputs remain sealed.

```bash
# Freeze the pilot protocol before starting parallel data workers.
.venv/bin/python -c 'from scripts.phystwin.sequential_reference import protocol; protocol()'
# Repeat with GPU/worker indices 0, 1, 2. Each worker handles its disjoint cases.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/phystwin/sequential_campaign.py --worker 0
# After all reference workers finish, render local RGB/depth and marker tracks.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/phystwin/v06_visual.py
# Run direct and RGB-D inference on separate available GPUs.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/phystwin/v06_analysis_runner.py --observations direct
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/phystwin/v06_analysis_runner.py --observations rgbd
# Preserve actions and audit predictions on a finer grid, on a free GPU.
CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/phystwin/v06_numerical_audit.py --size 97
# Requires the existing official checkout and isolated PhysTwin environment.
.venv/bin/python scripts/phystwin/install_v06_force_adapter.py
# Repeat object 0, 1, 2 on available GPUs. Fits both budgets and halves dt.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/phystwin/v06_phystwin_runner.py --object 0
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/phystwin/report_v06.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/phystwin/render_v06.py
.venv/bin/python -m pytest -q tests/test_sequential_fixture.py \
  tests/test_sequential_policy.py tests/test_v06_visual_bridge.py
```

The visual interface uses known cameras, geometry, and colored fiducial identities;
it is not raw-video reconstruction. PhysTwin gets 27 visible markers per frame,
so its three-object bridge is separate from the sparse, equal-budget policy
comparison. Added interactions also increase optimizer updates at the same epoch
count. PhysTwin learns spring parameters; the continuous physical posterior is a
separate FEM model. Videos distinguish static shapes and interpolated reference
ramps from validated dynamic ground truth. Full images, depth maps and simulation
fields remain local; only derived reports and example videos are published.
