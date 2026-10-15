# PhysGauss-SDE

Research prototype for physically constrained Gaussian scene reconstruction from **one moving camera with missing frames**.

The initial milestone is a small, reproducible CPU experiment. It is not a reproduction of PIDG or a completed real-video reconstruction system. We separate uncertainty over unknown scene parameters from stochastic forcing, and evaluate both against synthetic ground truth.

## Implemented first milestone

- Volumetric tetrahedral Neo-Hookean mechanics with fixed material mass and explicit supports.
- Gaussian primitives whose positions and covariances follow the material deformation.
- Known moving camera; deliberately irregular observation times and a contiguous missing interval.
- Image-based inference over stiffness, damping, and initial state.
- A correlated neural diffusion field, trained on an independent synthetic calibration task.
- Joint approximate inference over parameters and latent stochastic increments using observations on both sides of a gap.
- Held-out rendering, trajectory, uncertainty, and physical-consistency metrics.

## Setup

Use Python 3.10 or newer. Install the appropriate PyTorch build for your platform first if needed.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
```

```sh
physgauss check
physgauss demo --quick --output runs/smoke
physgauss demo --config configs/cpu_demo.json --output runs/cpu-demo
# Reuse matching fits and rebuild posterior samples/report:
physgauss demo --config configs/cpu_demo.json --output runs/cpu-demo --resume
```

Open `runs/cpu-demo/report.html` for an animated comparison, trajectory/uncertainty plots, and scores. The report is self-contained. The run also saves a configuration, JSON measurements, input observations, posterior trajectories, and a checkpoint. Source fingerprints prevent accidental reuse after a code change. Partial checkpoints can resume the remaining fit.

The default experiment uses 21 timestamps, six observed RGB frames, a 0.3-0.7 second observation gap, 24x24 images, and 64 posterior samples. All six input frames contribute to offline smoothing. This is deliberately small enough to inspect and run on CPU; it is not a high-resolution scene reconstruction.

`physgauss check` runs 12 numerical and physical checks with the standard-library test runner. GitHub Actions runs these checks and a small end-to-end demo.

## Research boundaries

The first benchmark assumes known rest geometry, density, camera poses, appearance, and material family. These assumptions deliberately isolate motion inference. Geometry completion, camera estimation, contact inference, and real-video fitting are later milestones. A Gaussian's spatial covariance represents material extent; posterior uncertainty is represented separately. The neural diffusion amplitude is trained using independent simulator-generated innovations, not inferred from a single video.

See [docs/METHOD.md](docs/METHOD.md) for equations, likelihood, physical units, and limitations of the posterior approximation.

## Project map

| Module | Purpose |
| --- | --- |
| `physics.py` | Volumetric elasticity, integration, supports, conservative internal impulses |
| `render.py` / `camera.py` | Deformation-coupled splatting and known pinhole camera paths |
| `stochastic.py` | Neural noise amplitudes and correlated spatial basis |
| `benchmark.py` | Separate observation and evaluation data; explicit scene priors |
| `inference.py` | Joint MAP smoothing and local Gauss-Newton Laplace approximation |
| `metrics.py` / `report.py` | Predictive scores, physical diagnostics, and experiment report |
| `cli.py` | Reproducible execution and checkpoint handling |

## Handover

See [HANDOVER.md](HANDOVER.md) for current status and [docs/RESEARCH.md](docs/RESEARCH.md) for the experimental plan. Development uses small pushed commits. Generated data and large checkpoints are excluded; the repository will include reproducible commands and a compact result summary.

