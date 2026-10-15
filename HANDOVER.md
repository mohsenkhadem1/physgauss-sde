# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

Milestone 1 is implemented, executed, and pushed. Twelve local tests and clean Linux CI pass. Two full CPU experiments are complete: stochastic truth and a matched deterministic control. Animated reports and measured summaries are versioned under `examples/`. Checkpoint resume has been exercised. This is a controlled synthetic proof of concept, not a completed real-video system.

## Start here

1. Read `README.md` for setup and run commands.
2. Read `docs/METHOD.md` for exact assumptions and the implemented equations.
3. Read `docs/RESULTS.md` for measured outcomes and their limitations.
4. Open `examples/cpu-demo/report.html` and `examples/deterministic-control/report.html` in a browser. They are self-contained.
5. Run `physgauss check` and the quick demo before changing the model.

The original implementation commits are `189f3d4` (project/protocol), `b6b535b` (mechanics/renderer), and `d35428d` (inference, metrics, reports, CI). Later result snapshots preserve the exact source fingerprint in JSON. The local isolated runtime is outside the repository; a new collaborator should create their own virtual environment using the README.

## Implemented

- Compressible Neo-Hookean tetrahedral FEM with fixed mass, clamped supports, gravity, and drag.
- Material-coupled 3D Gaussian centers and covariance; differentiable reference splatter.
- An observation-only RGB likelihood at six irregular times; no withheld targets in fitting.
- A positive neural diffusion amplitude with coherent low-rank spatial noise and central internal impulses.
- Independent simulator supervision for the diffusion network, explicitly disclosed.
- Joint scene/Brownian-increment MAP smoothing, local Gauss-Newton Laplace samples, and checkpoint resume.
- Position, rendering, CRPS, energy-score, coverage/width, and physical diagnostics.

## What the first experiments show

On stochastic truth, the SDE reduces the missing-gap coordinate RMSE from 2.794 to 1.340 mm in this realization. On deterministic truth it changes 1.221 to 1.244 mm and substantially widens intervals. Its 100% componentwise coverage is over-coverage, not calibration. The immediate priority is learning when process noise is needed, followed by independent seed/gap sweeps. Material identification is conditional on the known rest geometry, density, material family, and camera path.

## Decisions

- Start with fixed-topology deformable solids.
- Use genuine volumetric mechanics rather than independent splat motion.
- Separate Gaussian shape covariance, scene-parameter uncertainty, observation noise, and process noise.
- Use all available observations for offline smoothing; never access withheld images during fitting.
- Compare a deterministic physical model with parameter uncertainty against the same model with a correlated neural diffusion field.
- Label local Gaussian/Laplace approximations explicitly; they cannot establish multimodal posterior correctness.
- Do not redistribute the user's paper or credentials.

## Next milestones

1. Add hierarchical inference/shrinkage of process-noise scale; compare against the fixed calibrated diffusion prior.
2. Benchmark independent seed and gap sweeps, including deterministic and misspecified cases; assess coverage and interval width together.
3. Add inferred geometry and camera uncertainty.
4. Add contact events, stronger multimodal smoothing, and real videos.
5. Compare against PIDG, GP-4DGS, ODE-GS, and uncertainty-aware reconstruction methods under matched observations.

## Important limitations

- Rest geometry, appearance, camera path, density, material family and boundary supports are known.
- Eight nodes and six tetrahedra are only a minimal volumetric discretization.
- Diffusion is calibrated from separate synthetic innovations, not identified from one video.
- Local Gaussian inference cannot represent distinct contact/topology hypotheses; optimization has a finite budget.
- The present renderer is a dense CPU reference. There is no high-resolution CUDA reconstruction, real-video loader, learned camera, or geometry-completion pipeline yet.
- Model-selection comparisons need care: the SDE has more latent dimensions and an additional learned prior.
- No claim of state-of-the-art reconstruction or calibrated real-world uncertainty is supported by these two runs.

