# Campaign handoff — 2026-10-21

## Current campaign state — C24 published, C25 prepared

This section supersedes the historical publication, live-job and next-work notes below. Current main is C24 `76fa11fea7bc94d748f9351a69fb490a96236a10`. No Git refs were changed while preparing this update. Publish only reviewed source and compact UTF-8 reports; the user explicitly requires raw datasets, PT/NPZ files, videos, rational region tables and archives to remain LOCAL. Avoid binary/base64 payloads in tools or chat.

All earlier owned jobs are terminal: the 96-case finite batch, 96 independent linear predictions, six fixed Gaussian-network fits, and the completed known-support material-oracle diagnostics. The six-fit supervisor and all six child PIDs were verified absent; its session was reaped with exit0. The finite batch index is `runs/gaussian-campaign-v01/hypothesis-cohort-v01/result-index.json`, SHA256 `f79096a7c9eb7231d6810ccbcef95f9f3ebd9f701ecee879cb01358b8001f229`: 96/96 completed, 0 failed/queued/live, 5,856 saved frames. Do not duplicate or restart completed jobs. Unrelated machine workloads, including GPU3, remain outside campaign ownership.

Completed prediction result: 48 supplied-physics hypotheses (4 public hardware contexts ×4 BCs ×3 shared joint material tuples) ×2 globally identical rich force programs produce 96 engineering captures. Roles remain 32 fit /32 force-transfer /32 middle material. Same public/private 120 Hz and 61 outputs; IDs/private rates are excluded from model features. The six fits retain the same force-linear Gaussian architecture, width 64/layers 3, 4,000 steps, seeds20261022/23/24 and all final checkpoints. No seed/checkpoint selection, new architecture, residual model, retry or post-result tuning occurred.

Mean coordinate RMSE in mm (exchange / no exchange / supplied-physics linear): fit 0.1882 /0.1885 /0.01015; force-transfer 0.4687 /0.4640 /0.00924; middle material 0.5730 /0.5609 /0.02334. Physical linear mechanics wins decisively; Gaussian spatial exchange has no consistent advantage. No predictions invert in this window. Source/report inventory `runs/operator-dev/crossed-gaussian-training-v01/publication-inventory.json` freezes 33 files/454,463 bytes (SHA256 `1c0133c19a898143590b94645ff1c8d1c2500e9e790df6094aa7adbff1580b58`). Full per-BC/material/seed evidence: `runs/gaussian-campaign-v01/crossed-gaussian-training-v01/final-comparison/{summary.json,REPORT.md}`. Preserve all negative results.

The cohort is dependent: four reused hardware contexts, three joint material tuples, support states shared across materials, and only one new point-pin temporal refinement. This does not qualify every capture's temporal/continuum accuracy or establish four separate material effects. True material/support hypotheses are supplied. Seed spread describes optimization variation, not calibrated predictive uncertainty. Old native/production/calibration/probing gates stay unchanged, and reserved old response data remain unopened by this work.

C24 known-support oracle is separate: one fresh material draw, two sparse depth frames and an analytic sensor condition a four-parameter numerical material posterior. Relative to a proper-prior mean, future A error improves 0.370→0.113mm and held reset-force B 0.216→0.080mm. Known-parameter controls, prior-predictive Monte Carlo errors and limitations are in `docs/GAUSSIAN_MATERIAL_ORACLE.md`. One object does not establish coverage; this is not unknown-boundary inference or continuing-state active probing.

Active work now targets an unknown point-support mixture on H0. The geometry-only exact pushforward of the existing normalized point-pin prior has 55 discrete node masks, 473 positive atlas/Voronoi regions and exact total mass 1. Face probabilities remain 1/6 despite unequal eligible-cell counts. Its source/test/compact summary/design are frozen in `runs/operator-dev/point-support-mask-prior-v01/publication-inventory.json`; the full rational region table remains LOCAL. It uses no responses, observations, physics or posterior runs. It is preparatory geometry evidence, not an unknown-support prediction result.

Other agents own the packed boundary forward adapter and new unknown-support inputs. Those in-progress files are excluded from C25 until their owners freeze and review them. Current C25 scope is the completed six-fit comparison, frozen point-mask source/tests/compact evidence, README/docs/public summary and this latest handoff. No new simulations, architectures, archives or Git ref updates are part of this documentation/publication task.

## Historical notes — superseded by the current state above

## Latest continuation checkpoint (supersedes publication/progress details below)

Main is now `b0bb6f4f6e4d0f57bb839d07838e9f49b4e7110f`, tree `bf148d3b53a96a6f5f1fd03cd4e6ff65ba845e2e`. Published C4 `ce2b62e` (53 files), C5 `0fd3abb` (20 files), C6 `b0bb6f4` (19 files). Each remote commit was fetched and every intended blob compared against current local bytes before a leased ref update. Current campaign suite: 111 tests passed. Do not republish these checkpoints. Pending untracked sources belong to new PGS and free-deformable controls; preserve them.

New completed evidence:
- All six original/recentered 60/120/240 Hz cells pass full sensor QC but both temporal gates and translation gate FAIL. Original RMS 412/225/169 micrometres; recentered 285/233/47. `numerical-lowrate-v01/`.
- All four matching-rigid-iteration cells (32/64, original/recentered) pass QC and reproduce previous rigid16 trajectories bitwise. Native material tensor getters give intended E=4249.3676758 Pa, nu=.340718448; native rest points and oriented tets match all 12 reset branches. This verifies material-buffer values and SDK-consistent USD binding; a direct body-material-index getter was not found. No native nodal mass getter exists. `numerical-matched-rigid-v02/`.
- Independent finite-strain static reference completed in `nonlinear-fem-static-t0001-v02/`: exact finite rigid patches, seven load scales, full tangent SPD and independent reaction-force/moment audit. Nonlinear vs linear peak coordinate difference 0.423 micrometres (0.326%). Source/tests are published. This is NOT native equivalence, dynamics, continuum truth or a replacement dataset. Preliminary v01 preserved.
- Independent declared corotated-energy/body-KE proxies vs logged held-force translational work show a large discrepancy for TGS. They are NOT native total energy or conservation certificates. Sources and qualified summaries are published.
- PGS request original960/32 runs with full QC and native checks, response differs from TGS; RMS309 micrometres and much smaller proxy-energy mismatch. RecenteredPGS32 also completed: RMS300 micrometres; translation RMSE42.096 micrometres exceeds declared15.453 micrometre tolerance. Thus PGS32 still FAILS. Source/results `numerical-pgs-attempt-v01/`, `numerical-pgs-recenter-v01/`. Recenter wrapper not published yet.
- Native free1g rigid-handle control 120/960 x forceflagfalse/true completed. All corrected-clock momentum, mass/inertia/COM and zero-torque gates pass; maximum momentum error21.08 ppm, fitted impulse gains near1. No simple force-amplitude bug in FREE body. Flag changes position integration convention. Original analysis failures remain preserved: getters return nominal double dt, clock advances float32-rounded dt. Separate exact source-derived clock adapter uses measured per-step time and retains all100ppm gates, no physics rerun. `native-free-handle-v01/clock-analysis-v01/analysis.json` authoritative. Source/control/config/adapter/tests published.
- `campaign_rigid_inertia.py`: preparatory exact six auxiliary mass-point realization of physical rigid inertia, finite-rotation kinetic energy/linear/angular momentum identities, planar case and physical triangle inequality rejection; eight tests pass. This does not implement dynamics yet.

Archive phase09 is LOCAL ONLY: `runs/gaussian-recovery-phase09-archive/manifest.json`, 104,186,032 bytes, SHA256 `db9ea855eb5d3529040252fd9fbd08dca0a2855e80e7a6c184478baacb47377e`, 5,479 files. Clean restore `runs/restorechecks/gaussian-recovery-phase09` verified every file; restored linear and nonlinear sources/inputs replay all11 NPZ arrays bitwise. Proof `runs/gaussian-recovery-phase09-local-replay/proof.json`. Archive does not include later matched/PGS/free-handle controls (planned next archive). Do not call it remotely restored/published. An async question requesting local GitHub authentication is pending: no local credential helper, token, gh CLI or usable SSH key/agent key. Code publishes through connector; avoid bulk dataset payloads there because they inflated history. Keep researching independently; this does not block all goal work.

Current agents/tasks (last `/proc` check confirmed both listed drivers live):
- PGS64 original cell has since completed full QC: RMS308.905 micrometres, 32→64 RMSE3.966 micrometres; energy proxy essentially unchanged. Recentered64 child2069161 was live on GPU2, parent2068233. Do not infer the final translation outcome before its terminal analysis.
- Free-deformable v02 has now LAUNCHED: driver2069590 and first child2069592 live on GPU1, `native-free-deformable-v02/execution.json`. Supersedes the earlier not-launched statement below. Verify same handles; do not duplicate/restart because of a tool observation timeout.
- `/root/native_solver_diagnosis`: PGS64 original/recentered two-cell test, frozen new source/config `campaign_numerical_pgs_iterations*`, run `numerical-pgs-iterations-v01/`. Last known live driver2068233, first child2068346, tool session33026; VERIFY live handles and ledger before waiting/resuming. GPU2. No automatic128 or timestep sweep; only propose128 after evidence of decisive contraction.
- `/root/fem_review`: THREE free-deformable+same-two-handles inertial controls (TGS original/recentered and PGS original), GPU1. New `campaign_free_deformable_control*`, launched as v02 as noted above. Parent caught drag confound before launch: explicitly override linearDamping to0, preserve E/nu/rho and internal elasticity damping, remove fixture only, retain necessary native volume construction. Unforced zero branch uses common uniform .001m/s co-translation to check momentum leakage; diagonal starts rest. Infer body-mass proxy using rest-volume-weighted body COM velocity and known rigid masses/velocities, record assumptions and declared-mass momentum residual. Earlier prepared confounded protocol is preserved as unlaunched v01. No actual nodal mass claim.
- `/root/publish_diagnostic_update` finished C6 metadata and a read-only backup-options report `runs/operator-dev/nonlinear-reference-options-2026-10-21.md`. Installed Newton1.6.1 lacks required exact attachments; newer documented ones compliant. Recommended explicit float64 coupled finite-strain dynamic reference before Warp acceleration if native remains unreliable. Auxiliary mass representation requires careful velocity/projection/constraint-work and temporal-refinement validation; no dynamic solver was implemented in that report.

All original gates, reserved data and campaign objective remain intact. No known-force improvement, reliable physical inference, calibrated uncertainty or active-policy success has been established. The current turn made progress by publishing reviewed checkpoints, completing causal diagnostics and adding validated reference mechanics; the goal remains active. Prefer compact outputs, no dataset base64 in tool/chat payloads.

Read this file in a fresh chat instead of branching or replaying the large previous conversation. Repository: `/home/bayespc2/vscode/guassianSDE`, remote `mohsenkhadem1/physgauss-sde`.

## User intent and current interruption

Continue the authorized multi-agent research campaign: improve known-force deformation prediction, distinguish forward-model error from inference error and ambiguity, calibrate physical uncertainty against a matching-prior numerical posterior, then compare active probing against fixed/random controls. Use local CPU/GPU and push completed, reviewed work. Bird remains stopped. Preserve original sealed results and negative evidence. Do not claim convergence or intelligent probing success prematurely.

Latest user request was to diagnose recurring Codex app/session crashes. New experiments were held during diagnosis; the research goal remains active. Do not interpret this as permission to delete session history, kill unrelated processes, or restart the user's app. Check for live jobs before launching anything; the last owned numerical jobs had completed, and no subagents were live.

## Session diagnosis and prevention

The original rollout is 327,518,058 bytes; a related branched rollout exceeds 3.1 GB. Streaming inspection of the original found 1,912 records, including 120 records over 1 MB (314,717,896 bytes). Records containing binary-transfer markers account for 319,497,494 bytes. Large base64 artifact requests are a major source of history size. Do not pass dataset binaries through conversation/tool arguments again; prefer an authorized file-based transport. Do not print base64 or entire large manifests/logs.

VS Code renderer log reports the extension host unresponsive on 2026-10-21 at 15:56:10 local. Codex extension log contains hundreds of thread-history warnings about dropping items for an unknown turn on resume, plus UI ResizeObserver errors. No causal crash stack was found. History restoration/client overload is the leading hypothesis, not a confirmed root cause. Kernel checks found no OOM kills or GPU resets that day. Available RAM was 103 GiB on the final check; earlier it was 71 GiB. Separate desktop executor logs contain expired reconnect credentials; those can interrupt sessions but do not establish a UI crash cause.

Local evidence (sensitive logs; do not upload wholesale):
- `/home/bayespc2/.config/Code/logs/20261021T155556/window1/renderer.log`
- `/home/bayespc2/.config/Code/logs/20261021T155556/window1/exthost/openai.chatgpt/Codex.log`
- `/home/bayespc2/.codex/logs_2.sqlite` (read-only when inspecting)
- `/home/bayespc2/.codex/sessions/2026/10/20/rollout-2026-10-20T19-49-51-01a15aa6-045d-7fa2-9573-44689701e449.jsonl`

## Publication state

- Local/main and last fetched origin main: `f1168bfe376a3905baedcc0a2a0eac0a7125ad01` (recheck before pushing).
- Reviewed but UNPUBLISHED C4 checkpoint: `runs/campaign-c4-paths.json`, 53 explicit files totaling 1,672,852 bytes. `runs/prepare_campaign_c4.py` generated it. Recompute blob hashes before using it. Last targeted suite: 74 passing tests and clean diff whitespace check. New FEM and external-force flag evidence were outside that checkpoint and still need review.
- Working tree intentionally contains modified/untracked research sources, tests, reports and public evidence. Preserve them; never reset/clean. Read `git status --short`.
- Artifact branch: `gaussian-recovery-artifacts-2026-10-21`; last verified phase08 head `ead091400b78bb60847ab4731d1e6a65faa487cb`, tree `dc70eb143df4461a1ed8036b0ff780ff124a3a26`. Phases05–08 remotely restored and verified. Phase08 covers 12,374 files / 342,626,301 archive bytes. Proofs and artifact index updated locally in C4.
- New factorial/cross-device/capture/flag/FEM data are not archived remotely yet (planned phase09). Git push credentials were unavailable; connector Git-data API was used. Find a safe file-based upload route rather than resuming bulk base64 chat calls.
- Original `reports/gaussian-sealed-study-v01` and original artifacts index remain frozen. Actual/predicted video at b375eac remains evidence of interpolation, not future prediction.

## Scientific findings and gates

Known-force prediction has NOT passed G1. Best recent selected development fit (~0.478–0.506 mm) does not beat old controlled teacher (0.469 mm). Static ~0.641 mm. Reserved gate v8..15 and fresh confirmation cases must remain unopened predictively. Calibration-fit t112..119 was captured; calibration-audit t120..127 never started. Do not resume broad capture, calibration, or active-policy evaluation until the numerical reference is reliable.

Native Isaac reference currently fails numerical invariance:
- Fixed-patch TRAIN t1 coordinate RMS at 480/960/1920 Hz: 111/237/148 micrometres; adjacent differences increase 181→275 micrometres.
- Solver iterations and global translation substantially change response; high iterations can trigger compression guards. Recentered 32/64/128 RMS: ~40/153/886 micrometres.
- Crossed GPU assignment reproduces both original/recentered trajectories bit-for-bit: device assignment does not explain the discrepancy in this test.
- Removing per-sample rendering/readback/image writing gives bit-for-bit identical physics trajectory: capture backaction does not explain it in this test.
- External-force-every-iteration flag completed: true32/true64 RMS ~226/711 micrometres, true128 fails compression at 1.2 s. Flag alone does NOT fix the problem. Review `runs/gaussian-campaign-v01/numerical-external-forces-v01/analysis.json` and associated source/config/tests before publishing.
- Old 25D inputs omit full attachment patches/inertia. A paired same-input/same-force patch intervention changes trajectory ~7 micrometres (~30% signal). All future methods need the same full public hardware descriptor or explicitly model it as latent. Hardware audit v02 fixes ordered offset metadata; v01 preserved.
- Richer quadratic force response helps three development cases, worsens one; not a universal fix. Hard fixture projection alone improves total error <0.3%; capacity increase is not the first diagnosis.

Read `docs/GAUSSIAN_CAMPAIGN_NUMERICAL_AUDIT.md`, `docs/GAUSSIAN_RECOVERY_CAMPAIGN.md`, `examples/gaussian-neural-boundary/campaign-status.json` and reports `gaussian-campaign-numerical-v01/v02/v03` for evidence. Some public status predates final flag/FEM results and needs an explicit update after review.

## New CPU FEM sanity reference — review pending

`experiments/campaign_linear_fem.py`, `tests/test_campaign_linear_fem.py`, outputs `runs/gaussian-campaign-v01/linear-fem-t0001-v01/`.
455-node/1728-tet small-strain isotropic FEM, exact rigid attachment condensation and fixed constraints, explicit authored mass/inertia. Reduced stiffness SPD. Dynamic 240/480/960 Hz refinements decrease 16.8→4.2 nm; dynamic peak strain ~0.52%. This is a useful converged independent approximation, NOT validated equivalence to nonlinear PhysX or continuum truth. Review constitutive/damping assumptions and implementation, run meaningful tests, then decide how it diagnoses the native failure. Do not quietly substitute an easier linear problem for the user's deformable objective.

## Numerical posterior prototypes

`campaign_rejection_prior.py`: full hardware required, boundary category held fixed during geometry rejection; all 108 development TRAIN geometries supported. This is a NEW designed prior, not exact conditioning of the seed generator. BC-dependent unknown normalization constants mean `log_kernel` is not normalized `log_prob`; same-BC local moves and whole-prior independence proposals are supported. Rejection budget failures must abort the whole call, never silently redraw/drop a category.

`campaign_tempered_reference.py`: adaptive ESS-tempered SMC, resampling, local MH plus whole-prior refresh, explicit stratum contract and strict support masking. Tests include continuous/categorical analytic cases and rejection-prior integration. No actual calibrated video/material posterior demonstrated yet. Uniform final weights are not a mixing certificate; require multi-seed, particle-count and mode agreement. Read source/tests and `docs/CAMPAIGN_REFERENCE_PRIOR_DESIGN.md` before extending.

## Next work

1. Keep this session small; no bulk binary tool payloads. Preserve all disk outputs and log history.
2. Recheck C4 file hashes and validation, then publish the reviewed checkpoint with a lease on current main.
3. Review final external-force negative result and independent FEM source/tests; update status/report candidly and publish separately.
4. Repair or validate a nonlinear numerical forward reference before trusting physical inference or selecting probing actions. Keep prediction, calibration and active-policy gates separate.
5. Resume matching-prior numerical posterior comparisons and held-out force prediction only when the reference qualifies. Report positive and negative outcomes, and push reviewed work.
