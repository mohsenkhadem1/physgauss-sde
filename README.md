# PhysGauss-SDE

Research prototype for physically constrained Gaussian scene reconstruction from **one moving camera with missing frames**.

Current version: **0.6**, adding density-correct second-order staggered dynamics and continuous Gaussian depth-volume rendering, alongside force marginalization, structured force posteriors and nonlinear sampling. Reliable real-world uncertainty and real-video reconstruction remain open work.

The [accurate pipeline](docs/ACCURATE_PIPELINE.md) implements and tests both numerical fixes. Run `physgauss accurate --plan configs/accurate_reference_study.json --output runs/accurate-reference` for the new independent RK4-reference comparison, including component controls and five view schedules. Historical experiments remain versioned.

The [simulation and view audit](docs/SIMULATION_VIEW_AUDIT.md) found a material accuracy limit: the historical integration step has about **3.3 mm coordinate error** against converged independent RK4, despite consistent force formulas and momentum balances. Historical results are discretized-model experiments, not validated millimetre-level physical reconstruction. [Actual images, camera coverage and ten extra view-count fits](examples/simulation-view-audit/report.html), plus high-accuracy reference inputs, are committed. More observations help, but integration accuracy and renderer depth discontinuities also require attention.

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
# Matched control: ground truth has no process noise.
physgauss demo --config configs/cpu_deterministic.json --output runs/deterministic-control
# Reuse matching fits and rebuild posterior samples/report:
physgauss demo --config configs/cpu_demo.json --output runs/cpu-demo --resume
```

Open `runs/cpu-demo/report.html` for an animated comparison, trajectory/uncertainty plots, and scores. The report is self-contained. The run also saves a configuration, JSON measurements, input observations, posterior trajectories, and a checkpoint. Source fingerprints prevent accidental reuse after a code change. Partial checkpoints can resume the remaining fit.

The default experiment uses 21 timestamps, six observed RGB frames, a 0.3-0.7 second observation gap, 24x24 images, and 64 posterior samples. All six input frames contribute to offline smoothing. This is deliberately small enough to inspect and run on CPU; it is not a high-resolution scene reconstruction.

`physgauss check` runs 68 numerical and physical checks with the standard-library test runner, including integrator accuracy, the staggered density Jacobian, depth-tie continuity and multi-camera information identities. GitHub Actions also runs the original demo/study/audit/sampler checks, force-posterior and force-marginalization smoke experiments, and the accurate-reference pipeline.

## Analytically marginalized forces

Version 0.5 infers the solver-step trajectory and mechanics together, integrates the Gaussian force coefficients analytically, then recovers a force distribution including trajectory uncertainty. A matched comparison uses alternating force/mechanics updates, joint explicit-force optimization and marginalization under exactly the same trajectory model.

```sh
physgauss marginalize --quick --output runs/marginal-smoke
physgauss marginalize --plan configs/marginal_force_study.json --historical-summary examples/force-study/summary.json --output runs/marginal-force-study
```

Read [the derivation, covariance identities and comparison protocol](docs/FORCE_MARGINALIZATION.md). The new model declares full-rank free-node velocity diffusion; it differs from the historical simulator's absent or balanced low-rank neural noise. Force elimination is exact for this model. The trajectory posterior remains a local Gaussian approximation, and material/load identifiability remains unresolved. Historical simulator scores provide context rather than isolating marginalization's effect.

The [completed 12-fit comparison](docs/MARGINALIZATION_RESULTS.md) and [self-contained report](examples/marginal-force-study/report.html) are committed with exact input images, fitted paths and portable force distributions. Marginalized, joint and alternating solvers pass the local stationarity screen in 2/4, 1/4 and 0/4 conditions respectively. Accuracy is similar; no consistent advantage over joint optimization is established. Independent posterior-approximation diagnostics fail in the pulse and stochastic cases. [Depth-order step scans](examples/marginal-step-diagnosis/) expose visibility-related objective discontinuities; uncertainty remains provisional.

## Structured external-force distributions

Version 0.4 infers a smooth external load, fixed-amplitude internal SDE impulses, or both, through the observed RGB frames. Fifteen Gaussian-prior coefficients describe three loading directions over five temporal inducing points. Nodal loads use a known material-attached spatial profile; this is not arbitrary force-location inference. Outputs include force means, intervals, covariance and finite-window impulses, with explicit N/Ns units and convergence checks.

```sh
physgauss forces --quick --output runs/force-smoke
physgauss forces --plan configs/force_study.json --output runs/force-study
physgauss forces --plan configs/force_study.json --output runs/force-study --resume
```

See [the force prior, nonlinear target, diagnostics and controlled protocol](docs/FORCE_POSTERIOR.md). The full plan compares no load, a deterministic external pulse, internal stochastic forcing, and both, using the same images per model. There are no inferred model probabilities. A zero external-force band in the SDE-only model reflects an excluded component, not a conclusion that the force is absent. Inspect chain diagnostics before trusting any interval.

The [completed 12-fit experiment](docs/FORCE_RESULTS.md), [self-contained report](examples/force-study/report.html), and [portable posterior files](examples/force-study/cases/) are versioned. All twelve fits fail the full convergence screen at the tested budget. Force estimates and covariance are therefore provisional, despite passing software/numerical checks. Published files carry their diagnostic status and source/data provenance.

## Adaptive noise and independent-realization study

Version 0.2 adds an explicit zero-process-noise component and observation-only inference over noise scale. It analytically integrates scene parameters and stochastic increments under a common local image likelihood. Matched no-noise, fixed-SDE, and adaptive-mixture controls use the same inference approximation. Positive scale does not automatically mean physical randomness: the study also tests an unmodelled deterministic force.

```sh
physgauss study --quick --output runs/scale-smoke
physgauss study --plan configs/scale_study.json --output runs/scale-study
physgauss study --plan configs/scale_study.json --output runs/scale-study --resume
```

See [the adaptive-noise method and study protocol](docs/ADAPTIVE_NOISE.md). The full plan contains 18 cases: three independent seeds, two matched six-image schedules, and three ground-truth conditions. Coverage, interval width, CRPS, ranking metrics, scale-prior sensitivity, and linearization error are recorded. This remains a known-geometry experiment.

The [completed 18-case results](docs/SCALE_RESULTS.md) and [self-contained study report](examples/scale-study/report.html) are versioned. Adaptive inference suppresses unnecessary noise in deterministic cases and improves the stochastic cases in this small study. Unmodelled deterministic forcing still produces severe under-coverage and a nonzero inferred noise scale.

## Nonlinear likelihood audit

The local likelihood can be fragile even when average pixel error is small. The new audit evaluates independent proposals using the full nonlinear observed-image likelihood and records importance-weight ESS, concentration, physical failures, and source compatibility. It does not call reweighted intervals calibrated.

```sh
physgauss audit --study runs/scale-study --source-ref HEAD --output runs/likelihood-audit --samples 256
```

That command targets a fresh study from the current committed checkout. The historical milestone-2 raw local run requires `--source-ref d225760`. See [audit equations and provenance requirements](docs/LIKELIHOOD_AUDIT.md).

The [completed audit](docs/AUDIT_RESULTS.md) evaluated all 18 configurations with 256 proposals per method. Adaptive proposals fell below the predeclared relative-ESS threshold in **14 of 18 configurations**. The apparent accuracy gains therefore do not establish reliable nonlinear posterior uncertainty. The [audit report](examples/likelihood-audit/report.html) records the failures and their source provenance; a nonlinear smoother/sampler is the next inference milestone.

## Nonlinear Metropolis sampling

The `sample` command uses the full nonlinear physics/rendering likelihood for every Metropolis decision. The local Gaussian mixture preconditions pCN, independent, and scale-transport proposals. Its scale weights no longer determine the final posterior by themselves. Proposal adaptation stops after warmup; rank-normalized/folded split Rhat and bulk/tail ESS monitor every scene/noise coordinate, observed log likelihood, and adaptive process scale.

```sh
physgauss study --quick --output runs/new-local-reference
physgauss sample --study runs/new-local-reference --source-ref HEAD --output runs/nonlinear-smoke --warmup 20 --draws 32 --evaluation-samples 16
```

Use a clean committed source ref matching the study. For the historical local `runs/scale-study`, use `--source-ref d225760`. Full default settings use four chains, 1,000 warmup transitions and 2,000 retained transitions per chain. Add `--langevin` for gradient-guided pCNL with the full reverse-proposal correction. Add `--case stochastic__long_gap__3101` to select one exact case; repeat `--case` for more. Failed convergence diagnostics leave predictions explicitly provisional. See [the target, transition derivation, diagnostics and reproduction instructions](docs/NONLINEAR_SAMPLING.md).

The [selected long-gap experiments](docs/NONLINEAR_RESULTS.md) and [self-contained sampler report](examples/nonlinear-sampling/report.html) record deterministic, stochastic and missing-force cases with four chains each. They include convergence failures despite accurate-looking trajectories. This is a numerical investigation on one geometry and one realization seed, not a new calibration benchmark. The [next development sequence](docs/NEXT_EXPERIMENT.md) describes nonlinear proposal fitting, structured external forces, material viscosity, uncertain cameras and partially observed geometry; those extensions are not yet implemented.

## Historical first measured results

Both experiments ran locally and CI passed on Linux. In the stochastic synthetic realization, missing-interval free-node coordinate RMSE was 2.794 mm for uncertain deterministic mechanics and 1.340 mm with the neural SDE. In the deterministic control the results were 1.221 mm and 1.244 mm, respectively. The SDE intervals were substantially wider and over-covered the truth in both cases. This is a useful ablation, not a calibration or generalization claim.

See [complete measurements and interpretation](docs/RESULTS.md), the [stochastic report](examples/cpu-demo/report.html), and the [deterministic control report](examples/deterministic-control/report.html). Download/open the HTML reports locally; GitHub's file viewer does not execute them.

![Missing-frame comparison](examples/cpu-demo/preview.png)

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
| `scale_inference.py` | Integrated scale evidence and zero-noise/positive-noise posterior mixture |
| `study.py` / `study_report.py` | Matched controls, independent realization/gap sweeps, and aggregate reports |
| `likelihood_audit.py` / `audit_report.py` | Nonlinear likelihood overlap and reproducibility diagnostics |
| `mcmc.py` / `chain_diagnostics.py` | Nonlinear Metropolis transitions and multichain Rhat/ESS |
| `nonlinear_study.py` / `nonlinear_report.py` | Observation-only sampling, isolated predictive evaluation and trace reports |
| `forces.py` / `force_inference.py` | External-load GP projection and nonlinear fitted-proposal correction |
| `force_study.py` / `force_report.py` | Conditional force/noise comparisons, force/impulse distributions and reports |
| `metrics.py` / `report.py` | Predictive scores, physical diagnostics, and experiment report |
| `cli.py` | Reproducible execution and checkpoint handling |

## Handover

See [HANDOVER.md](HANDOVER.md) for current status and [docs/RESEARCH.md](docs/RESEARCH.md) for the experimental plan. Development uses small pushed commits. The repository includes source, configurations, tests, compact result snapshots, and self-contained reports. Generated raw data and large checkpoints are excluded and reproducible through the run commands.

