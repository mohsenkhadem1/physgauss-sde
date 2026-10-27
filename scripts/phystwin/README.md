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
