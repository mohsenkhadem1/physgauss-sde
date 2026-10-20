# Handover

## Active beam action selection — completed in this task

The user requested active/random/fixed action selection on the existing beam.
Implementation and equations: `experiments/beam_active.py`,
`experiments/run_beam_active.py`, `docs/BEAM_ACTIVE_DESIGN.md`.
Frozen config: `configs/beam_active.json`. All output and data are in
`examples/beam-active`; the README there explains interpretation and limits.

Local CPU completed 400 matched and 200 shorter-force-prior cases, with six
actions per policy. Active improves measurement efficiency: at two actions,
hidden RMSE is 0.805 mm vs random 1.649 and fixed 1.902. At six actions active
0.547 and fixed 0.554 mm nearly agree; 366/400 cases have identical final
measurement sets and posteriors. Final active 90% tip coverage is 93.25%, with
3.52 mm mean width. All declared matched scalar gates pass. Under the wrong
force correlation prior, active ends at 2.525 mm but only 27% coverage:
uncertainty is NOT robustly validated under mismatch.

Eleven beam unit tests and all 12,600 archived posterior replays pass.
`experiments/verify_beam_active.py` checks hashes, data pairing, budgets,
no repeats, posterior/metric replay, covariance PSD and action minimization.
The read-only `.github/workflows/beam-active.yml` reproduces on standard CPU.
No paid resources are needed. Earlier Bird and Isaac work is preserved.
The next scientific limitation is model-mismatch-aware action scoring and
comparison against an optimized fixed ordering; early gains here are against
a strong spatial/probe design with chronological time ordering.

## User-directed switch to Isaac Sim — 20 October 2026

User explicitly requested: stop Bird; create a simulated flexible block with
random fixed points and randomly applied forces using Isaac Sim. The Bird
coordinator, four workers and snapshot watcher were terminated. Raw saved
retained counts are4,150/4,100/4,150/4,100; checkpoint hashes are in
`runs/bird-local-production/user-stop-checkpoints.json`, execution status
`stopped_by_user`. All four complete4,000-draw stage snapshots remain intact.
Do not restart Bird or launch the wider comparison without a new instruction.

The active work is `experiments/isaac_boundary_block.py`, using the existing
Isaac Sim6.2 Python environment in the neighboring `deformable worldmodel`
project without modifying it. Known material, a random surface pin (optional
fixed patch), and Newton-valued random loads form the first forward simulator.
Loaded and unloaded explicit-clock episodes pass independent integrity replay.
Sleeping and settling damping are disabled; linear drag is explicit0.5/s.
The240/480Hz check remains sensitive (elastic RMS0.717/0.674mm, total
displacement231/183mm), so continuum/time convergence is not claimed.
See [setup and limitations](docs/ISAAC_BOUNDARY_BLOCK.md).
No boundary inference or new calibration result is claimed yet.
All83 stopped Bird files and461 Isaac development files are permanently
archived with roundtrip and Git blob hash checks on artifact commit
8605a4cbe35272d554185bf60568d355f3aae422. Compact results and raw pointers are
in `examples/isaac-boundary-block` and the Bird `raw-archives.json`.

## Earlier research discussion — 20 October 2026

The user's supplied discussion makes **boundary-condition learning** the main
scientific focus: infer the location, type, parameters and timing of supports,
fixtures, boundary tractions and contacts from a static Gaussian reference and
partial observations. Begin with known geometry/material and hidden-boundary
experiments. DINO/segmentation, material uncertainty and sequential interactive
updating are later stages. The [decision record](docs/BOUNDARY_LEARNING_DIRECTION.md)
records the agreed direction; the subsequently implemented beam reference is
documented in [the two-Gaussian formulation](docs/GAUSSIAN_BOUNDARY_FORMULATION.md).
The existing code does not yet infer general boundary patch/type hypotheses.

Historical status before the explicit stop instruction above: the bounded
Bird audit was running under its frozen target and source. The complete4,000-draw gate fails92/733 monitors: Rhat1.22643, minimum
bulk/tail ESS12.93/10.51, zero retained invalid/divergent steps and passing
density replay. Exact continuation to the declared8,000-draw cap is active.
The3,000-draw preview had already deteriorated, dominated by late motion even
though all five physical/noise parameters passed. Earlier improvement at2,000
was insufficient.
Do not claim convergence or open the wider comparison. The pasted discussion's
8,000-draw result refers to the older CPU run, not this local CPU/GPU attempt.

The separate constant-noise velocity-collapse prototype has three passing
density/derivative/geometry tests. It has no sampler recovery, mixing or
production evidence and is not integrated into the immutable active worker.

## Historical 2,000-draw local stage — 20 October 2026

The first full local2,000-draw stage fails45/733 monitors: worst Rhat1.012508,
minimum bulk/tail ESS256.58/131.34. Zero retained invalid/divergent state steps;
CPU replay and independent integral checks pass. Exact frozen continuations
are active to4,000, with8,000 available at the unchanged declared gate.
No new withheld-position scores or larger comparison have been opened.

Numerical worker remains4f0f0fcbfd518bd7be9b77001f5b23f34d5541dc. All four
RTX2080Ti GPUs render in float64; mechanics and RNG remain on CPU. A full
bridge preconditioner, joint/scalar parameter updates and deterministic CUDA
recovery preserve the physical posterior. All200 regression tests pass.
The remaining issues are early motion, late rigid-motion tails, mechanics
scale/logdet_Q and seven conditional load means. Stiffness, damping and the
old worst physical frame10/mode6 pass their scalar checks at2,000.

The independent conditional-transport256-draw pilot remains unconverged
(Rhat1.09703, min bulk/tail ESS33.16/85.34); it has no clear advantage and is
not promoted. Its states are discarded. Paired long-trajectory probes do not
justify changing the immutable full-run kernel. No physical/noise prior was
retuned and no pilot draws enter full estimates.

Live status: `runs/bird-local-production/execution.json`, `chain-*.log`.
A separate watcher preserves each complete stage checkpoint, including RNG
and counters. The original parent diagnostic ZIP, all initial development
states and the full2,000-draw failed stage are permanently preserved on
`bird-local-artifacts-2026-10-20` (latest stage commit
0ef4b691b10b7fdb24a878eb81a871f924080aa4). All file, archive and uploaded
Git blob hashes were independently checked. See the
[local audit](docs/BIRD_LOCAL_SAMPLER.md),
`examples/bird-local-sampler-audit/production-2000`, and `raw-archives.json`.

## Objective

Improve physical realism and calibrated uncertainty for Gaussian scenes reconstructed from one moving camera with irregular, missing observations.

## Status

**Current research focus: Gaussian boundary learning.** The user redirected the
work to a simple beam with hidden supports and loads, then proposed two Gaussian
families and conditional flow matching. `docs/GAUSSIAN_BOUNDARY_FORMULATION.md`
defines material fields, boundary laws, and the proposed conditional generative
model (generation time is distinct from physical time). The implemented first
benchmark fixes material properties and known 2D correspondences; it does not
train DINO, segmentation, or flow matching. It exactly enumerates24 boundary
configurations and analytically integrates smooth Gaussian force histories.
`examples/beam-boundary/report.html` contains1000 matched-prior cases and three
400-case mismatch stress arms, plus full compressed synthetic datasets.
Sparse/richer hidden-motion RMSE is2.727/0.627mm;90% missing-tip coverage is
89.5/89.8%. Exact boundary MAP identification is only19.3/32.9%. Known-boundary
oracle RMSE is2.682/0.536mm. The matched-prior scalar calibration gate passes;
shorter true load correlation causes severe undercoverage (49.5/15.75%), so
uncertainty is NOT validated under temporal-model mismatch. Separate stress
arms were added after the initial combined mismatch failure. Six numerical
tests pass, including independent cantilever and dense-conditioning references.
The study uses a40-element inference mesh and80-element reference, audited
against160 elements. A read-only free-CPU Actions workflow reproduces it.
The older GitHub CPU Bird run remains stopped and unconverged. The separate
local CPU/GPU audit described above is preserved under its own protocol; this
beam work does not modify it. Do not treat the beam result as a Bird fix.

**2026-10-20 final8000-draw result:** all four free-CPU chains completed.
Worst Rhat improved to1.0339864, minimum bulkESS134.503 and tailESS33.233.
The gate still FAILS (Rhat<1.01; both ESS>=400 for all monitored scalars).
Invalid/divergent retained state steps remain zero; density replay passes
with maximum cached-target error3.64e-12. Run39415918136 finished13:38:53UTC.
The bounded follow-up correctly stopped at the declared8000-draw cap, without
uncertainty evaluation or the wider comparison. That GitHub CPU run is inactive;
the separate local CPU/GPU audit is described at the top of this file.
The complete JSON convergence report and stop decision, retrieved from the
completed diagnose job log, are in `examples/pacnerf-bird-reflected-8000`;
its full checkpoint artifact12268611805 remains on GitHub. Further inference
audit is needed before trusting predictive uncertainty or expanding the matrix.

**2026-10-20 completed four-chain diagnostics:** mixing is improving, but Bird
has not converged. Worst Rhat fell from 1.87275 (256 draws/chain) to 1.33201
(2000) and 1.08886 (4000). At 4000 the minimum bulk/tail ESS is 39.08/11.49;
the strict gate remains Rhat <1.01 and both ESS >=400 for every monitored
quantity. There are zero invalid or divergent retained state steps. Full
convergence reports, per-quantity diagnostics and verified artifact provenance
are saved in `examples/pacnerf-bird-reflected-2000` and
`examples/pacnerf-bird-reflected-4000`. Run39408516882 completed computation
successfully but failed the statistical convergence gate. Exact continuation
run39415918136 targets8000 draws on free CPU workers; three of four workers
have finished as of this update. Four-chain diagnostics are pending. No
independent uncertainty evaluation or wider comparison has been launched.
Worker source remains ad3ec582aaa5e5ff1f4fdddf76318bd9d0bde031.

The earlier entries below describe preceding stages.

**2026-10-20 reflected/interwoven update:** recovery is proven; Bird convergence
is still pending. The mean-anchor floor-free guide preserves physical states
to3.79e-14. An exact centered parameter step now complements the noncentered
innovation update. Its nonlinear-funnel quadrature, actual neural joint/marginal
path-preservation, and bit-identical single-chain resume tests pass. The real
Bird16-draw integration check replays the target to3.64e-12. These are correctness
checks, not convergence evidence. The previous local950009 guide trial was
stopped during warmup; its50-step checkpoint is preserved, no draws pooled.

`.github/workflows/pacnerf-bird-reflected.yml` runs four independently seeded
standard free CPU workers,500 warmup/2000 retained draws, with archived256-draw
previews and a strict four-chain force/normalizer/state convergence gate.
The run record is `configs/pacnerf_bird_reflected.json`: active39399618548,
frozen workerad3ec582aaa5e5ff1f4fdddf76318bd9d0bde031. Preparation passed all188
tests on GitHub, rebuilt the shared reference and started all four CPU workers.
Exact continuation and independently gated uncertainty workflows are prepared
but not dispatched: `pacnerf-bird-reflected-continue.yml` and
`pacnerf-bird-reflected-validate.yml`. Use the recorded parent run and draw count.
The bounded follow-up is configured to extend failed full gates to4000/8000
or dispatch uncertainty validation on convergence. It pins the shared reference
SHA and source fingerprint, stops at8000 and never starts the wider matrix.
See `pacnerf-bird-reflected-followup.yml` and `experiments/route_reflected_bird.py`.
Partial/failed workers retain artifacts and require inspected exact recovery;
no infinite retry is scheduled. The separate16-draw four-shard integration
check passed density replay and correctly failed the production gate.
The new actual Bird256-draw GitHub checkpoint passed a separate local
32-versus16+16 continuation proof: samples, RNG, densities and counters are
bit-identical and the original prefix is unchanged. Proof in
`examples/pacnerf-bird-mixing-audit/reflected-real-checkpoint-recovery-proof.json`.
No local proof samples are pooled with the production chains.

**Latest observed-only preview:** all four workers completed500 warmup and256
retained samples and continued to2000. Full gate fails: worstRhat1.87275,
minimum bulk/tailESS5.688/11.186,518/733 monitored quantities fail, zero invalid
or divergent retained steps. The worst physical state is236 (frame10 first
elastic mode6). Stiffness/dampingRhat are1.063/1.062; latent motion remains the
bottleneck. New state-coordinate empirical covariance spans0.0614–74.41;
six straight physical interpolations have zero uphill excess, which does not
exclude entropic barriers, other paths or unvisited modes. No predictive scores
were opened. Evidence/plots in `examples/pacnerf-bird-reflected-preview`,
reproduction through `experiments/audit_reflected_preview.py` after aggregation.
The wider comparison remains gated; this preview is not a convergence result.
See `docs/PACNERF_REFLECTED_SAMPLING.md`. No larger comparison is dispatched
until Bird converges AND independently validated uncertainty meets readiness.
For exact continuation, use the original workflow source SHA and immutable
reference blob; script hashes are strict. Do not resume older prototype output
under the new runner/source fingerprint.

The following earlier status entries are historical.

**2026-10-20 recovery/mixing audit:** original run39214740152 is finished with
failure: first worker timed out after saving3350 draws/chain, recovery rejected
last-bit recomputed image correlations. Restoring saved values exactly makes
the unmodified worker settings guard pass and replays the target to3.64e-12.
See `docs/PACNERF_BIRD_RECOVERY_AND_MIXING.md`. All182 regression tests pass
in292.8s. Free CPU proof39391075353 passed:3350 original draws remain exact,
and32 extra draws match two16-draw continuations bit-for-bit, including RNG.
Protected original source files remain unchanged.

The local full joint-HMC trial finished256 draws and failed: Rhat1.656,
minimum bulk/tailESS6.45/13.15. The3350-draw audit identifies poor conditioning
(covariance eigenvalues0.011–91.99), slow mechanics/diffusion drift, and good
mixing of RGB/smooth-load scales. Tested physical interpolation paths show no
uphill barriers; global multimodality is not excluded. The late-reference
global elliptical trial finished256 draws and failed (Rhat3.412, minESS4.43).
The slice-augmented reflective sampler passes six new mathematical tests;
its actual Bird preflight passes. The first empirical-covariance reflective
trial was stopped in warmup because conditional directions still force small
steps. No convergence result is claimed. A floor-free conditional Gaussian
GUIDE (unchanged physical target) is running in `runs/bird-reflected-free-guide`,
log with`.log`, seed950009,128 warmup/256 draws planned. Its reference is
`runs/bird-floor-free-reference/reference.pt`. This learns only five parameter
moments; state covariance is the analytic conditional Gaussian bridge.
The guide roundtrip/density test passes for neural joint and marginal models
(183 tests total across the full182-test suite plus this new test).
See `docs/PACNERF_REFLECTED_SAMPLING.md`. All parent draws remain discarded
proposal-learning data; no future geometry has been scored.
The larger comparison has not started and must remain gated on Bird readiness.

The following historical entries do not override this audit update.

**LATEST: floor-corrected single Bird case, still under development.** All seven
original continuations finished:10 baseline cells converge, six Bird cells do
not. Single-Bird run39155024009/source33ec931 was canceled after an observed-
only physical audit found30–70cm floor penetration in development trajectories.
See `docs/PACNERF_FLOOR_CORRECTION.md` and `examples/pacnerf-floor-audit`.
No baseline continuation was canceled or changed.

The new single-Bird free CPU run is39214740152, frozen numerical sourcea14766e:
https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39214740152 .
It starts from the corrected floor target and a fresh discarded reference-learning
phase. Its convergence and uncertainty outcome are pending. The old canceled
run39155024009 is historical and must not be treated as this attempt.

The local learned-reference trial finished256 draws/chain after100 warmup:
worstRhat1.632, minimum bulk/tail ESS6.55/10.65. It did not resolve convergence
(preceding conditional trial:1.639,6.57/10.78). Both avoid penetration below-2cm.
See `examples/pacnerf-floor-audit/learned-reference-development.json`.
A full joint-HMC local diagnostic is now running with the same learned
reference,100 warmup,256 retained planned, seed590009,time1–4,acceptance0.9.
Its explicit experiment runner is `experiments/bird_joint_hmc.py`; it has no
future-geometry scoring or production eligibility. The active local exploratory
wrapper is `runs/bench_learned_joint_hmc.py`, output
`runs/floor-learned-joint-hmc-development`, log with the same name plus`.log`.
Do not change or cancel the formal run merely because a short development
trial is unconverged; its500-step warmup and final stopping plan differ.

The current separate target is `floor-conditioned-hierarchical-v2`: opaque
y=0 floor/camera visibility, plus an explicit2mm softened feasibility event
on later mechanical/rendering points. The normalized SDE/load priors are
specified before that event; force marginalization stays exact. The mode guide
includes boundary curvature and a full observed-Hessian fixed reference.
The default sampler now alternates exact conditional state HMC (time.6–1.8,
target acceptance.9) with five parameter elliptical slices. Exact parameter-
constant operators are cached only for the state derivative. Correlated Gaussian
moments, full-vs-partial physical gradients, and bit-identical warmup/retained
resumption tests pass. New numerical modules live in scripts; the frozen src
and four old inference files remain unchanged.

Independent floor-conditioned validation expanded the same seeded experiment
from2000 to10000 datasets:50/80/90/95% intervals cover50.39/79.84/89.78/94.62%.
Both earlier and expanded results are preserved. The nonlinear quadrature
sampler validation also passes with unit-scale reference starts. These are
small-model checks, not high-dimensional PAC convergence or real-data calibration.
The64-draw floor trial had no point below-2cm but failed convergence. A local
sampling-reference adaptation pilot is ongoing. Do not launch the matrix until a complete
Bird result passes strict diagnostics, raw replay, coverage/CRPS and physics
guardrails. No paid workers have been started.

The following entries record earlier stages and must not override this update.

The local conditional pilot has now completed256 draws per chain after100
warmup: Rhat1.639, minimum bulk/tail ESS6.57/10.78, no point below-2cm.
It is NOT converged or a reconstruction result. The sampled damping range is
far from the fitted proposal reference; an observed-only warmup reference
adaptation is the next inference experiment.169 regression tests pass. All
physics/model priors and final convergence thresholds remain fixed during it.

The next worker discards150 warmup+150 draws/chain to learn only a sampling
centre/covariance, regularized halfway to its initial white reference. Last
learning states map exactly into the new coordinates; final500-step warmup
and retained draws use a separate seed. Mode stationarity refers to the
unchanged fitted reference. Saved anchor, learning chains and final reference
allow independent NumPy replay. The single-Bird workflow includes one bounded
330-minute exact recovery worker; the full matrix is still not launched.
After publishing the independently replayed Bird report, the workflow dispatches
the larger free CPU comparison only if every readiness check passes. Downstream
source-hash checks reject any inference change since that Bird run.
The downstream matrix reuses the validated Bird result as case8 (no new
replicate claim) and runs seven additional observed-only mixing pilots before
the other15 full cells. Pilot workers have the same330-minute CPU bound as
full workers, with the usual final diagnostics unchanged.
All172 local regression tests passed in254.1s, including nondefault initial-state
resumption and independent NumPy reconstruction/tamper rejection of the learned
reference. No new Bird convergence or real-scene uncertainty claim follows from
these regression checks.

**Latest instruction: converge and validate one Bird case first.** Use only
this machine and free standard CPU workers. `pacnerf-bird-check.yml` runs a
complete case8 posterior (500 warmup, up to8000 retained per chain), independent
target/numerical/score replay and a single-case report. It gates the larger
comparison. `publish_pacnerf_bird_check.py` records readiness only when
strict convergence/replay pass, 90% intervals cover at least80% of this
development scene's withheld coordinates, and CRPS is no worse than the old
converged Bird case. These exploratory guardrails do not establish calibration
across independent objects. Only a passing report triggers the larger workflow.
150 local regression tests passed before adding this single-case controller.
The 64-draw development trial was not converged and is being extended with its
unchanged reference/target for diagnosis; it is not a scientific result.

Single Bird run **39155024009**, frozen source **33ec931**, was dispatched on
free standard CPU. Its URL/source are in `configs/pacnerf_bird_check.json`.
The full workflow now requires a passing Bird artifact from identical inference
files. Do not dispatch it before inspecting the single-case result.
The independent nonlinear quadrature/sampler test passed for both force arms:
`examples/nonlinear-force-sampler-validation/validation.json`. It tests a small
cubic observation/unknown-noise posterior, not PAC calibration. All152 local
regression/readiness tests pass.

**CURRENT (18 October): original batch had nine converged cells; all seven CPU
continuations were explicitly launched.** Five have finished: case0 passes
at 8000 draws (Rhat1.00295, bulk/tailESS1187/660), cases9/11/13/15 still fail
convergence at 8000. Cases12/14 are still running at this recorded check.
Their original run is 39100239747 and
their continuation IDs are in `configs/pacnerf_resumptions_round1.json`.
The original compact report is in `examples/pacnerf-exact`; all 33 original
input/result/checkpoint ZIPs are in the approved permanent public release.
Do not describe the original full run as still running.

The evaluation-only overconfidence audit is in
`examples/pacnerf-overconfidence-audit`. Projection removes only a small part
of the coverage failure; Torus interval widths are 31–46 mm versus 97–158 mm
errors, with spatially correlated image residuals. See
`docs/PACNERF_UNCERTAINTY_CORRECTION.md` for the new exploratory hierarchical
model and full protocol. The numerical src fingerprint and four frozen old
inference modules have not changed. New inference lives in separate scripts.
The independent 2000-dataset linear hierarchical mixture test passes, with
90% gap coverage 89.6–90.55%; this is not a PAC calibration claim.

`pacnerf-resumed-report.yml` publishes the combined baseline after all seven
authorized continuations finish and prefix/RNG/target checks pass.
`pacnerf-discrepancy.yml` tests the new density, runs eight observed-only pilots,
then sixteen fresh full comparisons with exact blocked elliptical slices in a
fixed Gaussian reference. A single bounded automatic recovery resumes unfinished
new cases. The workflow commits a separate snapshot after raw-state replay;
no source, observations or priors are changed during sampling. Full states stay
in 30-day Actions artifacts. No paid workers have been provisioned.

The first hierarchical workflow was **39143185132**, source **a575e18**:
https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39143185132 .
`configs/pacnerf_discrepancy_v1.json` records the frozen controller/source and
plan. 142 local tests passed before dispatch, and the eight short pilots passed
mathematical density checks, but their Rhat3.38–3.85/minbulkESS4.3 showed poor
mixing. The feasibility gate was too weak and allowed full runs to start.
That candidate was explicitly canceled; baseline continuations were untouched.
`examples/pacnerf-discrepancy-inference-audit` preserves its pilot proof,
cancellation, actual-data gradient and reference-curvature diagnostics.

The repair retains the same physical uncertainty model and uses a parameter-
dependent Gaussian state bridge. `pacnerf_conditioned_transport.py` conditions
linearized contact/noise/RGB mechanics on the sparse images at each material/
noise scale; the full nonlinear physical density and guide Jacobian correct
every proposal. `fit_pacnerf_conditioned_reference.py` builds two observed-only
guide linearizations. State blocks24 plus five scalar parameter blocks reuse
exact parameter/pose caches. Separable float64 Gaussian rendering is checked
against the original dense evaluator. The new pilot gate requires all-quantity
Rhat<1.1 and bulk/tailESS>=100 in 500/1000 retained draws, before any fresh
sixteen-case full experiment can start. A single hard Bird preflight runs
before the other seven pilot arms. New full runs must start fresh; canceled
candidate states are not reused as scientific samples. Final calibration and
reconstruction improvements remain unproven until that run finishes.

The following paragraphs retain the historical development record. The historical PAC
target is superseded. Auditing all sixteen saved prefixes exposed 90.885%
relative covariance eigenvalue error and 0.925% load-map error in Bird four-view
constant/marginal high-stiffness states. Refined 128/256 quadrature and an
independent matrix reference agree closely and confirm the defect. All ten
continuations in `runs/pacnerf-recoveries-v2` were stopped; their checkpoint
artifacts have been collected. The completed historical cells below must not
be presented as corrected-model results or an overall scientific ranking.

The exact integrals are now active in `src/physgauss/pac_integrals.py` and
PACModel. The corrected numerical fingerprint is
`eb703c41c125e971edae446bcad13d8f80a0ca5a34efb9f6a7b156762bf9f655`.
It uses balanced 5x5 mode-pair exponentials for full covariance and 3x3 input
integrals. All 24 PAC-specific tests pass locally, including high stiffness,
high damping, gradients/Hessians, marginal/joint and innovation identities,
and checkpoint/RNG controls. A new 32-condition initial-geometry audit covers
both scenes/noise types through stiffness 9 and passes independent matrix
references. The old 256 saved-state prototype check gave errors <=2.57e-12
covariance and 6.65e-13 load. Old source is preserved in the detached worktree
`../physgauss-legacy-audit` at 2f07671; set PYTHONPATH to its src when replaying
old data, since the local venv is editable against this primary checkout.

Generic recoverable sampling lives in `src/physgauss/checkpoint_hmc.py`.
The first exact-integral Bird pilot is run 39098175068 at e77276e. Its
constant/marginal reference failed stationarity before sampling; neural/marginal
finished 100 retained draws, which are development only. This pilot predates
the corrected full batch described above.

A data-informed exact-innovation reference is now implemented in
`scripts/pacnerf_innovation_metric.py`. The full RGB Gauss-Newton metric is
assembled via exact pose sensitivities; nonlinear targets remain exact.
The observed-only difficult Bird case fitted successfully locally, decrement
6.30e-5, without future geometry. 27 PAC-specific tests pass, including density,
Jacobian, gradient, metric finite differences and checkpoint continuity. The
new worker records coordinate/module hashes and diagnoses physical plus native
states. Fresh pilot/full directories are mandatory. The old exact physical
coordinate pilot cannot resume into this worker. See
`docs/PACNERF_EXACT_PROTOCOL.md`. No priors, weights, observations or mechanical
parameters were retuned. The 118-test CI at e77276e passed before these new
inference changes; the new CI run 39099244484 passes all 121 tests and smoke pipelines.
The four-arm exact-innovation pilot is run 39099258939, controller 9573274,
actual frozen worker 219665b. Collector session 10645 writes
`runs/pacnerf-exact-innovation-pilot`; the corrected inputs are in
`runs/pacnerf-exact-inputs`, with all four PT files byte-identical to original
production inputs. The initial exact physical-coordinate pilot 39098175068
finished with three successes and the constant/marginal fit failure; all raw
artifacts are local in `runs/pacnerf-exact-pilot`. The exact-innovation pilot completed all four arms. Raw cache/density checks
pass (<=1.01e-11), acceptance 83-96%; short-chain Rhat/ESS fail full convergence
as expected. The conservative pilot screen is **false**, because neural/joint
has one rejected +181.38 energy-error proposal (1/400); three arms pass it.
This is preserved in `examples/pacnerf-exact-pilot`. It does not promote any
pilot to production. The planned fresh 500-warmup/2k-8k runs can now test mixing
without changing the model, priors or data. The full sixteen-case comparison is now run **39100239747**,
https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39100239747 .
Controller source is a0b048086946a5675a0eda0ecd6b79e445ae8095; every inference
job checks out frozen worker 219665b0776d9a1b5d092176331e21f724499bca.
Dispatch record is `runs/pacnerf-exact-production-dispatch.json`; no pilot chains
are reused. Collector writes `runs/pacnerf-exact-production`. All sixteen cases
start fresh with 500 warmup and 2k-8k retained draws, four chains per case.
The final job at the dispatched controller independently verifies the artifacts,
publishes `complete-pacnerf-comparison`, and normally pushes only
`examples/pacnerf-exact` plus `docs/PACNERF_EXACT_RESULTS.md` to main.
Do not edit that pending results document while this dispatched finalizer is
running: it deliberately refuses to overwrite concurrent changes. README and
handover can advance independently. If a job times out, its warmup/draw prefix
is uploaded; exact-resume checks out the same worker and requires the source
run. If publication fails, the artifacts/report remain inspectable; fix/report
with a new controlled publisher rather than altering frozen worker chains.
All sixteen case workers have entered computation on GitHub Actions CPUs.
Controller CI run 39100207840 passes **125 tests** plus the smoke pipelines.
Three corrected worker results are available; thirteen worker jobs are still running. The complete comparison and aggregate verification remain pending.

The publication controller now verifies checkpoints/scores/density identities,
both geometry and saved-state transition integrals, and pair agreement before
ranking. `finalize_pacnerf_exact.py` passed a local empty-batch end-to-end smoke
without falsely claiming numerical validity or ranking. The full Actions
workflow will run this finalizer after workers finish, upload its report and
commit compact artifacts using a separate disposable worktree with no force
push. Raw checkpoints never enter Git. It refuses an existing corrected report
or concurrently changed results document. Four publication/credential
safety tests pass locally, including an exact high-stiffness floor-only audit. The report controller may advance; worker and
numerical model stay pinned.

**Historical approximate-target milestone: six of sixteen final
worker results are available, five pass finite checks. Ten original workers reached their computation cap
with independently replayed retained prefixes (03:47 UTC, 18 October).** Read `docs/PACNERF_RESULTS.md`,
`docs/PACNERF_PROTOCOL.md`, and `examples/pacnerf-sparse/report.html`.

The complete six-view Torus four-arm regime is independently verified:

| Variant | Gap coordinate RMSE | Nominal 90% coverage | Draws/chain | Gradient batches / worst common pose ESS |
|---|---:|---:|---:|---:|
| Constant, marginalized | 97.46 mm | 17.84% | 4000 | 22.38 |
| Constant, explicit joint | 97.41 mm | 17.64% | 4000 | 51.03 |
| Neural, marginalized | 97.54 mm | 18.10% | 2000 | 19.67 |
| Neural, explicit joint | 97.48 mm | 17.90% | 4000 | 46.40 |

All four pass finite diagnostics and raw target/score/full force-moment replay.
Marginalization uses about 2.28/2.36 times less gradient work per worst common
pose ESS in this one regime; this is not a general hardware speed ranking.
Neural noise has no demonstrated physical accuracy or calibration advantage.
The neural marginal/joint pair has two modal mean coordinates just beyond the
approximate within-pair MCSE screen (max 3.64 vs cutoff 3.60); record this
borderline result. The complete planned eight-pair screen is still unavailable.
Four-arm force curves nearly overlap but are restricted acceleration profiles,
not validated measured forces. Their uncertainty is not force calibration.

Case 2 (four-view Torus neural/marginal) also passes: 81.48 mm error, 13.32%
coverage, 2000 draws/chain. Case 0 (constant/marginal) FAILS at 8000 draws/chain:
Rhat 1.3783, minimum bulk/tail ESS 8.73/76.54. It passed at interim 1000 then
failed at 2000/4000/8000; motion traces show late regime switches despite mixed
material parameters. Do not extend it beyond the declared maximum or rank it.
All six final cases replay numerically; numerical replay is not convergence.

**Frozen production provenance:** Actions run 39073611351, worker/model commit
`b2d081a7a55ecf1027921f5d841c043e23713a9d`, numerical fingerprint
`052a4cb2013105f9880912d3ab71b009dd9d65140a2ff44d2c30cab43d1ae1f4`.
Matching original CPU validation run 39073594746 passed 102 tests and the
historical smoke pipelines. Later reporting/checkpoint/innovation tooling
preserves physical source; full CI at 40e578e and 681e7ce passed 109 tests
(runs 39091917289 and 39092504961).

Raw data: `runs/pacnerf-data`. Production prepared inputs:
`runs/pacnerf-production-inputs`, artifact 12115787202, ZIP SHA256
`9b8249b6c64736f5a522cfa326edb31725d2599faf8da2623df6ccdc22ed4954`.
Original raw results/checkpoints: `runs/pacnerf-production`.
The original collector completed; all sixteen result/chain artifact pairs are
local. Ten workers reached the 330-minute cap around 03:37 UTC (confirmed in
completed job logs). Their retained prefixes are 1000/2000/4000 draws per chain.
Every saved prefix passes exact-target replay. Never count incomplete chains as
valid posteriors. `inspect_pacnerf_attempts.py` records attempt status, mode,
retained prefix, cached-target replay and computation-step timing separately.

For necessary continuation, `pacnerf-resume.yml` checks out b2d081a and fetches
the SHA256-pinned checkpoint wrapper from commit
`e9c09a0b3bb6e0a1670c08aeefb51224d642bff1`. It preserves the original target,
frozen adapted kernel/RNG and 2000/4000/8000 gates; checkpoints warmup every 25
transitions and retained blocks <=50. Four nonlinear controls reproduce the
original sampler stream exactly, including warmup interruption. No recovery
run succeeded yet. The first ten continuations (ledger
`runs/pacnerf-recoveries/dispatches.json`) stopped at the original worker settings
guard before sampling. All ten preserved original chain bytes exactly. The
identified bug is JSON list/tuple comparison in observed pairs and nested initial
appearance metadata. A container-only adapter and sixteen-draw preflight now
have two regression controls, alongside the four existing stream tests; all six
pass locally. Actual preflight 39095026025 now succeeds on Bird four-view
neural/marginal: 2000 unchanged original draws plus sixteen new draws, no new
warmup, unchanged reference/settings, exact RNG-consumption replay, and all 64
new cached targets reproduced within 5.69e-9. Evidence and provenance are in
`examples/pacnerf-resume-preflight`. The new six-control code passes full CPU CI
at e9c09a0/e59af8d (111 tests; runs 39094868532/39094962834).
Those continuations were relaunched in `runs/pacnerf-recoveries-v2`, then
cancelled after the confirmed integration defect. Do not resume this historical
quadrature target again; activate the exact-integral correction and freeze fresh
posterior runs. Case 10 had sourced the verified 2016-draw preflight prefix.
Source, input, environment and numeric
version fields remain strict; no physics or source-worker edits are made.
Use a fresh collector directory per recovery run
and preserve original artifacts/prefixes. A resumed worker's seconds cover only
that invocation; accumulate attempt records before discussing total time.

**Separate observed-only innovation feasibility pilot:** Actions 39092510895,
source `681e7ce969032c99956f071d9a59996d127914a6`, four Torus four-view arms,
identity prior reference, 100 warmup/100 retained draws, compute cap 25 minutes.
The pilot collector completed into `runs/pacnerf-innovation-pilot/case-i`.
All four hit the 25-minute cap during warmup: last recoverable transitions
25/50/50/50 at the .015 step-size floor; no retained posteriors exist. Completed
logs confirm each timeout. Warmup cache/density replay passes to 1.19e-11 /
8.19e-12. Compact negative evidence is `examples/pacnerf-innovation-pilot/summary.json`.
It reads no evaluation geometry and computes no reconstruction/coverage scores.
It cannot replace production evidence. See `docs/PACNERF_INNOVATIONS.md`.
The exact causal Gaussian-force innovation transformation passes three math
unit tests plus sixteen density/round-trip probes per completed case; maximum
density discrepancy <=1.23e-11 across all six cases. This establishes algebraic
equivalence, not better mixing. Do not change the frozen physics or weights.

**Scope and physical limits:** two official synthetic elastic simulations
(Torus/Bird), 14 timestamps, sparse schedules labeled 4/6 include frame zero
and supply 3/5 later dynamic images. Supplied initial geometry and eleven-camera
initial appearance bootstrap are explicit conditions. This is a twelve-mode
linear elastic graph with smooth frame-held contact, not original MPM or
unknown-geometry monocular reconstruction. Fixed cameras are subsampled; this
is not continuous moving-camera capture. Joint/marginal arms use the same
Gaussian load prior and have identical integrated motion posteriors in theory.
Future geometry is used only after sampling for scoring/reference audits.

Independent transition checks at sixteen probes per six final cases give
relative covariance eigenvalue error <=1.39e-7. However the nominal zero-noise
contact audit finds 16.41/17.20 mm whole-sequence differences against a refined
solve of the same reduced ODE. Even true supplied geometry leaves rendering
error, including missing table-depth occlusion/shadows. Both are material
limits. The frozen neural diffusion is transferred from independent cube
calibration; PAC calibration is not claimed. On sampled six-view marginalized
Torus paths the fixed floor contributes median 94.0–94.1% of one-step neural
position process variance; median neural/constant variance ratio is only
1.0075–1.0078. This local diagnostic excludes loads and full posterior variance.

Stopped development runs 39071805396, 39072191439 and 39072691871 are not
scientific evidence. Old `runs/pacnerf-inputs` and `runs/pacnerf-refined-development`
are not the frozen production inputs/results. The verifier uses the producer's
float32 Torch linspace interval IDs, then NumPy quantiles; float64 linspace
selected two different IDs at 32000 draws. Production scores were unchanged.

**Public raw archives explicitly approved, 18 October 2026.** The user replied
“Yes—publish the public checkpoint archive” and chose “Continue with free CPU
workers.” The previous automatic-review block is resolved by that explicit
human approval. No paid GPU should be provisioned for this study.

`scripts/archive_pacnerf_checkpoints.py` now separates `historical` run
39073611351 (b2d081a) from `exact` run 39100239747 (219665b). Both default to
local plan-only verification. Explicit publication uploads original verified
Actions ZIPs, checks server SHA256/size/state, never replaces or deletes assets,
and rejects mismatched study provenance before authentication. Historical
assets carry a prominent SUPERSEDED integration-defect label and have their own
prerelease. Corrected inputs/checkpoints use a different release and source tag.

`.github/workflows/pacnerf-checkpoint-archive.yml` supports a manual initial
archive and a completion hook restricted to this exact existing run ID. It
collects available completed artifacts, appends immutable release assets and
commits only the small public download/integrity manifest under `archives/`.
The source run's report finalizer stays frozen; do not edit its pending
`docs/PACNERF_EXACT_RESULTS.md`. Six archive tests pass locally (offline plans, wrong provenance, changed ZIPs,
refusing conflicting assets, initialization without upload, and CI refusing
missing releases). Initial archive job 39103518452 failed with GitHub HTTP 403
when trying to create the source-pinned prerelease. The existing owner
credential has now initialized `pacnerf-exact-2026-10-18` without uploading
assets. CI uses `--require-existing-release`, so it only appends verified assets
and cannot create source tags. The comparison is unaffected. Retry archive job **39103728525** uses the
fixed workflow; initial job 39103518452 is historical failure evidence.
The historical upload completed all **33 original ZIPs (500,448,430 bytes)**.
A public API read verified every remote asset ID/name/SHA256/size/state and the
source commit. Its manifest is `archives/pacnerf-historical-quadrature-2026-10-18.json`.
Release: https://github.com/mohsenkhadem1/physgauss-sde/releases/tag/pacnerf-historical-quadrature-2026-10-18 .
The corrected release is initialized at
https://github.com/mohsenkhadem1/physgauss-sde/releases/tag/pacnerf-exact-2026-10-18 ;
its prepared-input ZIP is now published and verified using the existing owner
credential while the retry job waits for a free runner. The manifest is
`archives/pacnerf-exact-2026-10-18.json`. The completion hook is configured to
append remaining case ZIPs for source run 39100239747; Archive job 39103728525 has completed successfully. Owner uploads and the
existing-release workflow have succeeded; remaining case assets will be added
by the completion hook.

Creating the two source tags unexpectedly triggered fourteen older push
workflows (path filters did not prevent these tag runs). Their metadata are in
`runs/archive-tag-triggered-workflows.json`. The initial cancellation attempt
was rejected by automatic approval review. The user subsequently explicitly
replied: “Cancel only the duplicate tag-triggered runs.” After that approval,
all fourteen exact run IDs were rechecked against their tag, source SHA, push
event and original creation timestamp. **All fourteen had already completed**;
no cancellation request was needed or sent. The main experiment 39100239747
remained in progress and was untouched. The audit record is
`runs/archive-tag-cancellations-approved.json`. No cleanup approval remains
pending. No paid GPU was started. These duplicate tag-triggered runs are not
new scientific results for the corrected PAC-NeRF comparison.

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

