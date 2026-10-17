# Handover

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

**Latest completed milestone: nonlinear posterior inference audit and 36-case rerun.**
Read `docs/POSTERIOR_RESULTS.md`, `docs/POSTERIOR_INFERENCE.md`,
`docs/POSTERIOR_CONVERGENCE_PROTOCOL.md`, and `examples/converged-posterior/report.html`.
All 36 benchmark cases pass every finite-chain diagnostic (56,172 scalar monitors;
maximum Rhat 1.00934, minimum bulk/tail ESS 820.9/555.8). Four chains per case,
500 warmup, 2,000 retained per chain; only `stochastic__4102__floor` extends to
4,000. The replacement uses exact-target Gaussian-reference split HMC with
cheap image Taylor proposal forces and full nonlinear image acceptance.
Physical equations, original RGB, noise weights, force marginalization, priors,
reference data and all normalizers are unchanged and reverified.

Stochastic-case MAP → posterior mean gap errors are **4.847→5.025 mm floor,
3.281→3.349 mm constant, 3.623→3.330 mm neural**. The neural–constant difference
of 0.019 mm is not evidence of superiority. Neural averaging improves two of
six stochastic cases and worsens four. Raw posterior central-component coverage
is 66.1/95.8/96.6%; correlated benchmark coordinates do not establish calibration.
The older 19+20-motion conformal study remains separate and unchanged.

The calibration-only control reruns old constant/old neural/oracle-amplitude
MAP under the final operator: on three pulse+noise seeds, calibrated/oracle
neural errors are 3.622/3.606 mm, versus legacy neural 3.160 mm. A refit from
the sampled mean returns to the original high-error mode. Density-volume
effects, not residual calibration error or this optimizer tolerance, explain
the development-case point-estimate failure. Model discrepancy still matters.

Implementation commit `6a00d90c0ca8a53183c8346fc28def808c8e26ec`, numerical
fingerprint `7d5b5c85b57a45819136acb4657b8b7c5041c0e145de12da0fb6683331984c27`;
later scripts/docs preserve it. **94 checks** pass. Main raw comparison:
`runs/converged-posterior-ci`, Actions run **39062694272**. Independent fixed-small-step
replicas: run **39063241871**. Portable calibration-only control: **39063531997**.
The initial neural replica has a borderline two-coordinate batch-MCSE difference;
one fixed extension of both kernels to 4,000 draws per chain (run **39065071522**)
gives 0/84 beyond the same approximate cutoff (max |z| 3.157 <3.434). Both
longer runs pass all monitors, with zero invalid/large-energy proposals; their
cached exact targets agree to 5.5e-12 and old prefixes are preserved bitwise.
These supplementary results do not replace the original benchmark or establish
calibration. Keep both `replica-comparison-initial.json` and `-extended.json`.
`scripts/converged_sparse_comparison.py` reuses published exact data/coordinates,
rebuilds its fixed Fisher proposal and verifies the old objective. Sampling
continuations preserve RNG and frozen kernels. `publish_converged_posterior.py`
verifies portable means, scores, every diagnostic flag, force covariance and
audit states before publication. Full raw latent chains/energy errors are
30-day Actions artifacts; permanent compact joint draws, summaries, all-coordinate
diagnostics and exact inputs support reproduction. The interrupted redundant
local long sampler is not evidence; complete CPU-worker runs are primary.

Next prioritize new-seed posterior calibration/predictive checks, floor/force-basis
and time/mesh sensitivity, then unknown geometry/cameras and real video. Finite
diagnostic passes do not prove global exploration or a true physical model.

### Previous calibration milestone

Version **0.8.1** is the preceding calibration experiment. Start with `docs/CALIBRATED_NOISE_RESULTS.md`, `docs/CALIBRATED_NOISE_METHOD.md`, `docs/CALIBRATED_NOISE_PROTOCOL.md`, and `examples/calibrated-noise/report.html`. All **114 fits** pass stationarity and their saved factors/targets reproduce locally: 36 matched benchmark fits plus 38 interval-calibration and 40 fresh-test fits. Neural amplitude error is **0.295% on fresh test states**. Paired stochastic-case gap errors are **4.847 / 3.281 / 3.623 mm** for floor/constant/neural noise; accurate calibration still does not show a neural-specific reconstruction gain.

Whole-gap prediction sets target 90% coverage using 19 independent calibration motions. On 20 fresh test motions, coverage is **19/20 constant** and **20/20 neural**, with mean component widths **21.70 / 26.28 mm**. Treat this as conditional synthetic prediction-set evidence, with wide Wilson intervals; it is not Bayesian posterior validation. The separate four-chain exact-target audit fails **all 1,548 monitored quantities** (worst Rhat 1.302; minimum bulk/tail ESS 10.5/18.3). Force conditionals remain exact at a fixed path, but their full inferred posterior remains provisional. Do not promote its MCMC averages/intervals into validated estimates.

Model source is `edbf12a4a872de1fdd1f7be4bd5ba687a7fe50c6`, fingerprint `7a4e3ce81327156803e359031e7365f001c275607b7f8026c997078b33769f96`; all **87 checks** pass. An initial cross-runtime recovery failure exposed a rank cut through a tied RBF eigenspace. The final source freezes the actual original v0.7 cube projection. A separate renderer correction implements the proper zero-extinction emission limit and its first/second derivatives. Initial summaries/failure evidence remain in `examples/calibrated-noise-diagnostics`; their preliminary coverage is superseded. Final held-out seeds are **83001–83020**, with no model retuning after examining initial test outcomes.

Primary raw artifacts are `runs/calibrated-noise-published`; the final posterior audit is `runs/calibrated-posterior-optimized`. Complete comparison and validation run in separate CI partitions, sharing serialized prior initialization and frozen calibrated weights. Original RGB is unchanged for paired fits; validation calibration forms a barrier before fresh test jobs. The local fixed-comparison pilot and earlier calibration pilots are not mixed into final cases. `scripts/materialize_validation_precisions.py` rebuilds factors from portable states without optimization, including optional disjoint CPU partitions. `scripts/publish_calibrated_noise.py` verifies and publishes all results. Raw matrices/chains are excluded from Git, with exact inputs, weights, states, diagnostics, posterior moment summaries, CSV, figures and hashes retained for regeneration. The optional `.[performance]` extra and audit settings document numerically checked contraction acceleration.

Next scientific priorities are nonlinear posterior exploration and motion/force/material ambiguity, then broader scenes/views and real-video/unknown geometry. Increasing network capacity is not supported by the present comparison.

### Historical status notes

Version 0.7 combines frozen simulator-calibrated neural internal diffusion with exact external-force marginalization. The **36-fit sparse-view comparison is complete**: start with `docs/NEURAL_MARGINAL_RESULTS.md`, `docs/NEURAL_MARGINALIZATION.md`, and `examples/neural-marginal/report.html`. On stochastic condition/seed cases, mean gap errors are 4.85/2.89/2.97 mm for floor/constant/neural internal noise. Constant captures the observed gain as well as neural adaptation. All 36 fits pass stationarity; local weight screens pass 12/12 floor and only 2/12 each added-noise model, so their local uncertainty is not validated. The evaluation-only calibration audit in `examples/neural-calibration-audit/` finds 36.3% neural amplitude error on true frame states and about 1.2% amplitude variation, versus 14.4% broad-feature validation error. No network or fit was retuned after this finding.

Numerical source is `08c3281`, fingerprint `689baa8640bcd18883d087d1a65f57feb11afd32e446c0250ac926fb71b25695`; all 77 checks and expanded CI passed. The proper state-dependent covariance/log determinants and information derivatives are implemented; conditional force covariance is averaged over inferred trajectories. `configs/neural_marginal_study.json` fixes all models/conditions/seeds. Raw local results are `runs/neural-marginal-final`. Compact publication includes exact inputs, reference audits/positions, calibrated network weights, actual spatial basis/constants, solver states, posterior force functions, primary and explicitly secondary diagnostics, all seeds and hashes. `scripts/recover_neural_posterior.py` reconstructs a local covariance without optimization and reproduced an original neural precision factor exactly. Independent stochastic reference and production refinement gates pass. The retained full-rank floor makes this a different model from the historical singular neural SDE. Next address operating-region calibration and nonlinear posterior validation before increasing neural/model flexibility.

Version 0.6 implements the two audit fixes: a second-order staggered integrator with its parameter-dependent density Jacobian, and permutation-invariant continuous depth-volume rendering. The full **20-fit rerun is complete and published**: start with `docs/ACCURATE_RESULTS.md`, `docs/ACCURATE_PIPELINE.md`, and `examples/accurate-reference/report.html`. Integration coordinate RMSE is about 0.130 mm against RK4-96 (maximum node error 0.626 mm); corrected six-view gap RMSE is 1.01/1.14 mm for no-load/pulse. All ten corrected marginalized view fits pass stationarity; both alternating fits fail. The four stationary primary fits pass the local importance-weight screen, but pulse results are borderline and uncertainty remains uncalibrated.

Implementation source is `c35b5a1`, fingerprint `fe2687d262d450cd8bf8f7127a3c1fe010c9df14c3e243f36eb80d98f7764702`. All 68 checks and expanded implementation CI passed. The fixed comparison is `configs/accurate_reference_study.json`; `physgauss accurate --quick` exercises the entire path. Raw checkpoints are in `runs/accurate-reference-final`; compact exact inputs, solver states, point/force distributions, CSV, plots and source hashes are published. Reproduce publication with `scripts/publish_accurate_study.py` and the implementation ref. It verifies all 20 fits, exact RGB/noise, initial states, saved objectives/paths and force covariance. The main study uses two deterministic references only; stochastic-reference validation, mesh convergence, posterior refinement and repeated-realization calibration remain open. Historical measurements below are preserved.

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

The density-correct higher-order integrator and continuous visibility surrogate are now implemented (v0.6). Six substeps give about 0.130 mm coordinate RMSE and 0.626 mm maximum node error against RK4-96. The state dimension is 1,448. Next establish fitted-posterior time refinement, mesh convergence, repeated-realization calibration, and an independently validated stochastic reference; the deterministic integration check alone does not establish these.

0. Validate the nonlinear trajectory posterior under continuous visibility, including multiple initializations and independent proposals/samplers. Test full-rank discrepancy sensitivity and incorporate balanced internal noise before expanding spatial force flexibility.

1. Establish adequate nonlinear sampler exploration on sparse-view cases using the implemented convergence/ESS diagnostics; do not equate a correct Metropolis kernel with converged finite chains.
2. Improve exploration of the implemented structured-force/SDE models under the corrected numerical pipeline. Only then extend spatial force patterns, model probabilities, unknown geometry/camera and repeated-realization coverage studies. Use sparse block linear algebra for larger all-at-once trajectories.
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

