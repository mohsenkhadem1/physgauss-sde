# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

Milestones 1 and 2 plus a nonlinear likelihood audit are implemented, executed, and pushed. Twenty-six local tests and the expanded Linux CI pipeline pass. The original two experiments, a matched 18-configuration study, and a 54-method/configuration nonlinear audit have versioned reports. This remains a controlled synthetic proof of concept, not a completed real-video system.

## Start here

1. Read `README.md` for setup and run commands.
2. Read `docs/METHOD.md` for exact assumptions and the implemented equations.
3. Read `docs/SCALE_RESULTS.md` and `docs/AUDIT_RESULTS.md` for the latest outcomes and limitations. `docs/RESULTS.md` records the historical two-case milestone.
4. Open `examples/scale-study/report.html` and `examples/likelihood-audit/report.html` in a browser. They are self-contained. Earlier animated reports remain under `examples/cpu-demo/` and `examples/deterministic-control/`.
5. Run `physgauss check` and the quick demo before changing the model.

The original implementation commits are `189f3d4` (project/protocol), `b6b535b` (mechanics/renderer), and `d35428d` (inference, metrics, reports, CI). Later result snapshots preserve the exact source fingerprint in JSON. The local isolated runtime is outside the repository; a new collaborator should create their own virtual environment using the README.

Latest implementation/evidence commits: `d225760` (adaptive noise and common-linearization study), `d495a69` (18-case measurements), and `6bc029d` (nonlinear likelihood audit and preserved observation data). Source fingerprints distinguish historical measurements from the current implementation.

## Implemented

- Compressible Neo-Hookean tetrahedral FEM with fixed mass, clamped supports, gravity, and drag.
- Material-coupled 3D Gaussian centers and covariance; differentiable reference splatter.
- An observation-only RGB likelihood at six irregular times; no withheld targets in fitting.
- A positive neural diffusion amplitude with coherent low-rank spatial noise and central internal impulses.
- Independent simulator supervision for the diffusion network, explicitly disclosed.
- Joint scene/Brownian-increment MAP smoothing, local Gauss-Newton Laplace samples, and checkpoint resume.
- Position, rendering, CRPS, energy-score, coverage/width, and physical diagnostics.
- A spike-and-slab process-noise scale prior, integrated Gaussian evidence, and joint scale/scene/forcing samples.
- Matched no-noise, fixed-SDE, and adaptive-mixture controls under one common local likelihood.
- Three independent realization seeds, two matched six-image schedules, and deterministic/stochastic/unmodelled-force conditions.
- AUSE, Spearman uncertainty ranking, interval coverage curves, and predeclared scale-prior sensitivity.
- Full nonlinear observed-image likelihood audits using independent importance proposals and ESS diagnostics.
- Source-compatible checkpoint auditing and persisted observation tensors/network weights for new studies.

## What the first experiments show

The historical first stochastic realization improved from 2.794 to 1.340 mm with fixed SDE forcing, but intervals were too wide on deterministic truth.

The new matched study scores all withheld frames, not only the largest gap. On stochastic truth with long gaps, mean coordinate RMSE over three seeds is 3.132 mm (no process noise), 2.373 mm (fixed SDE), and 2.004 mm (adaptive mixture). Adaptive noise largely vanishes on correctly specified deterministic truth and avoids unnecessarily broad intervals. Unmodelled deterministic forcing still yields a nonzero inferred noise scale and severe under-coverage.

Critically, the nonlinear likelihood audit flags adaptive-proposal relative ESS below 0.1 in 14/18 configurations. The local posterior is often fragile even when average pixel linearization error is small. Do not claim calibrated nonlinear uncertainty or treat the current reweighting diagnostic as a corrected posterior. Material identification remains conditional on known geometry, density, material family, appearance, supports, and camera.

## Decisions

- Start with fixed-topology deformable solids.
- Use genuine volumetric mechanics rather than independent splat motion.
- Separate Gaussian shape covariance, scene-parameter uncertainty, observation noise, and process noise.
- Use all available observations for offline smoothing; never access withheld images during fitting.
- Compare a deterministic physical model with parameter uncertainty against the same model with a correlated neural diffusion field.
- Label local Gaussian/Laplace approximations explicitly; they cannot establish multimodal posterior correctness.
- Do not redistribute the user's paper or credentials.

## Next milestones

1. Implement an iterative nonlinear smoother or a nonlinear posterior sampler, with convergence/ESS diagnostics. This takes priority after the likelihood audit.
2. Distinguish structured external forcing/model discrepancy from stochastic forcing. Repeat coverage and proper-score studies on independent realizations and more geometries.
3. Add inferred geometry and camera uncertainty after establishing reliable inference on the known scene.
4. Add contact events, multimodal hypotheses, and real-video ingestion.
5. Compare against PIDG, GP-4DGS, ODE-GS, and uncertainty-aware reconstruction methods under matched observations.

## Important limitations

- Rest geometry, appearance, camera path, density, material family and boundary supports are known.
- Eight nodes and six tetrahedra are only a minimal volumetric discretization.
- Diffusion is calibrated from separate synthetic innovations, not identified from one video.
- Local Gaussian inference cannot represent distinct contact/topology hypotheses; optimization has a finite budget.
- The present renderer is a dense CPU reference. There is no high-resolution CUDA reconstruction, real-video loader, learned camera, or geometry-completion pipeline yet.
- Model-selection comparisons need care: the SDE has more latent dimensions and an additional learned prior.
- No claim of state-of-the-art reconstruction or calibrated real-world uncertainty is supported by these synthetic experiments.

## Reproduce the latest pipeline

```sh
python -m pip install -e .
physgauss check
physgauss study --plan configs/scale_study.json --output runs/new-scale-study
physgauss audit --study runs/new-scale-study --source-ref HEAD --output runs/new-audit --samples 256
```

Use a clean committed checkout for `--source-ref HEAD`. The historical raw local run `runs/scale-study` requires `--source-ref d225760`; it predates the audit and did not persist observations. Its audit verified the matching numerical environment, forward reference render, and stochastic derivative probes before regenerating them. Current studies persist their observed images and diffusion weights.

Historical source changes deliberately invalidate ordinary `study --resume`. The audit is a separate operation that verifies compatibility with a committed historical source. Neither command silently mixes changed source, changed study measurements, or changed inference settings into existing results.

