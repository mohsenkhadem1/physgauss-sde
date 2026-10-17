# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

Version 0.6 implements the two audit fixes: a second-order staggered integrator with its parameter-dependent density Jacobian, and permutation-invariant continuous depth-volume rendering. Start with `docs/ACCURATE_PIPELINE.md`. The new fixed comparison is `configs/accurate_reference_study.json`; `physgauss accurate --quick` exercises the entire path. Integration tests compare directly to independent RK4-96 trajectories. The main study uses two deterministic references only; stochastic high-accuracy reference validation remains open. A full rerun will be published separately from historical evidence.

The simulation/view audit is complete: start with `docs/SIMULATION_VIEW_AUDIT.md` and `examples/simulation-view-audit/report.html` for actual native images, camera coverage, independent integration and ten matched view-count fits. Mechanics/constraint checks pass, but the historical three-substep integrator differs from converged RK4 by 3.3 mm coordinate RMSE (up to 11.5 mm node error). Preserve older results as discretized-model investigations; they do not establish millimetre-level physical accuracy. New deterministic high-accuracy inputs are published. All 60 local checks pass, including multi-camera AD equivalence and a <0.25 mm accuracy gate for the audited 48-substep case. Only 5/10 extra view fits pass local stationarity; more data improves information without resolving integration error, mismatch or hard depth sorting. Source `38be8cb` and manifest/script hashes identify the audit.

Version 0.5 adds exact Gaussian load elimination in a nonlinear trajectory model and a matched alternating/joint/marginalized solver experiment. Read `docs/FORCE_MARGINALIZATION.md` before using it: a declared full-rank velocity-discrepancy process is shared by the three new solvers, so this is not a reparameterization of the old singular low-rank simulator. Nine new tests check normalized Gaussian densities, conditional force moments, total covariance, Schur equivalence, nonlinear derivatives, causal density normalization and reproduction of the old solver with zero added innovation. Run `physgauss marginalize --quick`; the fixed full comparison plan is `configs/marginal_force_study.json`. Exact force elimination does not imply calibrated trajectory uncertainty or global nonlinear convergence.

The 12-fit comparison is complete: read `docs/MARGINALIZATION_RESULTS.md` and open `examples/marginal-force-study/report.html`. Numerical source is `e8face7`, fingerprint `9ce4a6f377460fbe0f5378ab3da500da3e6de68ebaaa905c3523e32dd6132a25`; all 57 checks and expanded Linux CI pass. `5701c07` corrects units labels only and adds reproducible publication/step-diagnosis scripts; numeric executable ASTs are unchanged. Marginalized/joint/alternating solvers pass local stationarity in 2/4, 1/4 and 0/4 conditions. No consistent accuracy or timing advantage over the joint solver is shown. Weight concentration and stochastic-case coverage remain poor. `examples/marginal-step-diagnosis` documents depth-order changes and objective jumps even under extremely small steps. Stabilize the visibility likelihood before claiming reliable nonlinear inference.

Raw local checkpoints: `runs/marginal-force-study`, with fit-boundary resume verified. Compact publication includes exact RGB inputs and shared initial states, each fitted solver-step path, conditional/Rao-Blackwell force moments and provenance. Full Gaussian precisions and paired joint draws remain in raw checkpoints or are regenerated with the fixed plan. The publication script verifies source, executable ASTs, input hashes, numerical records and force covariance. Use a new output directory to run with current source; the historical fingerprint intentionally rejects changed-source resume.

Version 0.4 is implemented and its 12-fit structured-force experiment is executed. Forty-eight local checks and the expanded Linux CI pass. The new API reconstructs coherent external-force functions, cross-time covariance and nodal samples from compact posterior files. All twelve controlled fits fail the complete convergence screen at four chains with 1,000 retained states each; estimates remain provisional. Original physics/SDE experiments, the 18-case scale study, likelihood audit and version-0.3 sampler investigation remain versioned historical evidence. This is a controlled synthetic proof of concept, not a completed real-video system.

## Start here

Version 0.4 adds a 15-coordinate structured external load, force-only/SDE-only/combined controls, nonlinear fitted proposals and explicit force/impulse distributions. The suite contains 48 checks, including normalized target/reference density checks, known Gaussian posterior recovery, physical image gradients, passive truth/impulse logging, rejection of indefinite nonlinear curvature and reconstruction of portable joint force distributions. The controlled study uses the stable Gauss-Newton proposal; full nonlinear likelihoods still determine acceptance. Start with `docs/FORCE_POSTERIOR.md` and `physgauss forces --quick`. These restricted known-profile posteriors do not infer arbitrary force locations, camera or geometry.

Read `docs/FORCE_RESULTS.md` and open `examples/force-study/report.html` for the latest measurements. Portable files under `examples/force-study/cases/` include an `inference` status/provenance block. Use `evaluate_force_posterior` from `forces.py` to query them, and keep joint `posterior.pt` coordinates when generating scene samples rather than independently combining force and material marginals. Raw local results are in `runs/force-study-v2`; a new clone can run the fixed plan into a fresh directory. Independent execution partitions were checked for identical source, environment, configurations and calibrated network tensors; `execution.json` records the merge.

Implementation commits: `480648f` (force model/inference), `1644958` (portable coherent posteriors), `e301b5a` (test-only cross-platform derivative check away from depth-order discontinuities). The full study's source fingerprint matches `1644958` and is unchanged by the test-only commit. CI for `e301b5a` passes, including all original smoke paths and the force pipeline.

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

Establish a sufficiently accurate inference integrator against the new independent RK4 reference inputs; avoid shared coarse simulation concealing bias. At 48 substeps the current all-at-once dimension rises to 11,528, requiring sparse methods or a density-correct higher-order formulation. Stabilize visibility before extending physical identification or calibration claims.

0. Resolve visibility/depth-tie numerical discontinuities exposed by the version-0.5 matched experiment; repeat its fixed comparison and validate a nonlinear trajectory posterior. Test full-rank discrepancy sensitivity and incorporate balanced internal noise before expanding spatial force flexibility.

1. Establish adequate nonlinear sampler exploration on sparse-view cases using the implemented convergence/ESS diagnostics; do not equate a correct Metropolis kernel with converged finite chains.
2. Improve exploration of the implemented structured-force/SDE models; compare gradient-free proposals and inspect renderer depth-order/linearization effects. Only then extend spatial force patterns, model probabilities and repeated-realization coverage studies.
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
physgauss forces --plan configs/force_study.json --output runs/new-force-study
physgauss study --plan configs/scale_study.json --output runs/new-scale-study
physgauss audit --study runs/new-scale-study --source-ref HEAD --output runs/new-audit --samples 256
physgauss sample --study runs/new-scale-study --source-ref HEAD --case stochastic__long_gap__3101 --output runs/new-nonlinear --langevin
```

Use a clean committed checkout for `--source-ref HEAD`. The historical raw local run `runs/scale-study` requires `--source-ref d225760`; it predates the audit and did not persist observations. Its audit verified the matching numerical environment, forward reference render, and stochastic derivative probes before regenerating them. Current studies persist their observed images and diffusion weights.

Historical source changes deliberately invalidate ordinary `study --resume`. The audit is a separate operation that verifies compatibility with a committed historical source. Neither command silently mixes changed source, changed study measurements, or changed inference settings into existing results.

