# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

Milestones 1 and 2 plus a nonlinear likelihood audit are implemented, executed, and pushed. Version 0.3 adds nonlinear Metropolis sampling with pCN and optional gradient-guided pCNL, defensive mixture/transport moves and rank/folded Rhat plus bulk/tail ESS. Thirty-seven local tests and the expanded Linux CI pass, including nonlinear quadrature and actual physics-likelihood gradient checks. The original two experiments, matched 18-configuration study, 54-method/configuration likelihood audit, and selected three-case nonlinear experiment have versioned reports. None of the three full sampler cases passes every diagnostic at four chains and 2,000 retained draws each. This remains a controlled synthetic proof of concept, not a completed real-video system.

## Start here

Version 0.4 adds a 15-coordinate structured external load, force-only/SDE-only/combined controls, nonlinear fitted proposals and explicit force/impulse distributions. The suite contains 48 checks, including normalized target/reference density checks, known Gaussian posterior recovery, physical image gradients, passive truth/impulse logging, rejection of indefinite nonlinear curvature and reconstruction of portable joint force distributions. The controlled study uses the stable Gauss-Newton proposal; full nonlinear likelihoods still determine acceptance. Start with `docs/FORCE_POSTERIOR.md` and `physgauss forces --quick`. These restricted known-profile posteriors do not infer arbitrary force locations, camera or geometry.

1. Read `README.md` for setup and run commands.
2. Read `docs/METHOD.md` for exact assumptions and the implemented equations.
3. Read `docs/NONLINEAR_RESULTS.md`, `docs/SCALE_RESULTS.md` and `docs/AUDIT_RESULTS.md` for outcomes and limitations. `docs/NONLINEAR_SAMPLING.md` derives the new transition kernel. `docs/RESULTS.md` records the historical two-case milestone.
4. Open `examples/nonlinear-sampling/report.html`, `examples/scale-study/report.html` and `examples/likelihood-audit/report.html` in a browser. They are self-contained. Earlier animated reports remain under `examples/cpu-demo/` and `examples/deterministic-control/`.
5. Run `physgauss check` and the quick demo before changing the model.

The original implementation commits are `189f3d4` (project/protocol), `b6b535b` (mechanics/renderer), and `d35428d` (inference, metrics, reports, CI). Later result snapshots preserve the exact source fingerprint in JSON. The local isolated runtime is outside the repository; a new collaborator should create their own virtual environment using the README.

Latest implementation/evidence commits: `d225760` (adaptive noise and common-linearization study), `d495a69` (18-case measurements), and `6bc029d` (nonlinear likelihood audit and preserved observation data). Source fingerprints distinguish historical measurements from the current implementation.

Nonlinear sampler implementation: `b30fcff`, version 0.3. Its 37 checks and expanded Linux CI, including fresh-study gradient-guided sampling, pass. The selected full-run investigation uses the historical `d225760` references, four chains per case and one realization seed. Read its convergence flags before using the predictive numbers. `docs/NEXT_EXPERIMENT.md` gives the ordered route towards more realistic physics and partial-view geometry/camera inference.

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
- Full nonlinear observation-only Metropolis corrections, per-chain warmup adaptation, explicit invalid-proposal rejections, and multichain diagnostics for every latent coordinate.
- Gaussian-reference pCN and gradient-guided pCNL with the full proposal-density correction, independent refresh and scale transport.

## What the first experiments show

The historical first stochastic realization improved from 2.794 to 1.340 mm with fixed SDE forcing, but intervals were too wide on deterministic truth.

The new matched study scores all withheld frames, not only the largest gap. On stochastic truth with long gaps, mean coordinate RMSE over three seeds is 3.132 mm (no process noise), 2.373 mm (fixed SDE), and 2.004 mm (adaptive mixture). Adaptive noise largely vanishes on correctly specified deterministic truth and avoids unnecessarily broad intervals. Unmodelled deterministic forcing still yields a nonzero inferred noise scale and severe under-coverage.

Critically, the nonlinear likelihood audit flags adaptive-proposal relative ESS below 0.1 in 14/18 configurations. The local posterior is often fragile even when average pixel linearization error is small. Do not claim calibrated nonlinear uncertainty or treat the current reweighting diagnostic as a corrected posterior. Material identification remains conditional on known geometry, density, material family, appearance, supports, and camera.

The nonlinear sampler's target is now corrected mathematically, but finite-chain exploration remains insufficient: 4/190, 122/190 and 190/190 monitored quantities fail the screen in deterministic, stochastic and missing-force long-gap cases respectively. All use realization seed 3101, so these are selected numerical investigations, not new independent calibration replicates. The stochastic scale diagnostic passes while other coordinates remain flagged; the missing-force case has severe chain disagreement. Read `docs/NONLINEAR_RESULTS.md` before using any displayed interval. Next priorities are nonlinear proposal refitting and an explicit structured force/discrepancy model, rather than increasing the flexibility of the SDE alone.

## Decisions

- Start with fixed-topology deformable solids.
- Use genuine volumetric mechanics rather than independent splat motion.
- Separate Gaussian shape covariance, scene-parameter uncertainty, observation noise, and process noise.
- Use all available observations for offline smoothing; never access withheld images during fitting.
- Compare a deterministic physical model with parameter uncertainty against the same model with a correlated neural diffusion field.
- Label local Gaussian/Laplace approximations explicitly; they cannot establish multimodal posterior correctness.
- Do not redistribute the user's paper or credentials.

## Next milestones

1. Establish adequate nonlinear sampler exploration on sparse-view cases using the implemented convergence/ESS diagnostics; do not equate a correct Metropolis kernel with converged finite chains.
2. Distinguish structured external forcing/model discrepancy from stochastic forcing. Repeat coverage and proper-score studies on independent realizations and more geometries.
3. Add inferred geometry and camera uncertainty after establishing reliable inference on the known scene.
4. Add contact events, multimodal hypotheses, and real-video ingestion.
5. Compare against PIDG, GP-4DGS, ODE-GS, and uncertainty-aware reconstruction methods under matched observations.

## Important limitations

- Rest geometry, appearance, camera path, density, material family and boundary supports are known.
- Eight nodes and six tetrahedra are only a minimal volumetric discretization.
- Diffusion is calibrated from separate synthetic innovations, not identified from one video.
- The nonlinear posterior still uses a fixed-topology model without contact hypotheses. Finite MCMC runs need convergence checks; a correct invariant target is not evidence that a run explored it adequately.
- The present renderer is a dense CPU reference. There is no high-resolution CUDA reconstruction, real-video loader, learned camera, or geometry-completion pipeline yet.
- Model-selection comparisons need care: the SDE has more latent dimensions and an additional learned prior.
- No claim of state-of-the-art reconstruction or calibrated real-world uncertainty is supported by these synthetic experiments.

## Reproduce the latest pipeline

```sh
python -m pip install -e .
physgauss check
physgauss study --plan configs/scale_study.json --output runs/new-scale-study
physgauss audit --study runs/new-scale-study --source-ref HEAD --output runs/new-audit --samples 256
physgauss sample --study runs/new-scale-study --source-ref HEAD --case stochastic__long_gap__3101 --output runs/new-nonlinear --langevin
```

Use a clean committed checkout for `--source-ref HEAD`. The historical raw local run `runs/scale-study` requires `--source-ref d225760`; it predates the audit and did not persist observations. Its audit verified the matching numerical environment, forward reference render, and stochastic derivative probes before regenerating them. Current studies persist their observed images and diffusion weights.

Historical source changes deliberately invalidate ordinary `study --resume`. The audit is a separate operation that verifies compatibility with a committed historical source. Neither command silently mixes changed source, changed study measurements, or changed inference settings into existing results.

