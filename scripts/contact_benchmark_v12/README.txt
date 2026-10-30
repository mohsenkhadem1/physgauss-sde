Object-centric/contact extension campaign v12
===========================================

All v11 published evidence and source are read-only. New runs live under
runs/contact-benchmark-v12 and publication artifacts under
reports/contact-benchmark-v12. Raw datasets, assets and checkpoints remain local.
The absolute campaign deadline is configs/contact_benchmark_v12.json. A future
reproduction must declare a new deadline and separate output directory.

Representation and scope
------------------------
Frozen Neo-Hookean material FEM supplies deformable node trajectories. A proper
rigid decomposition separates object pose from local deformation. Position-only
prefix calibration fits a pose correction and bounded local residual. Candidate
selection uses a separate earlier-prefix FEM fit and a recalibrated earlier
grasp; the full allowed prefix is used only after that choice. Initial geometry
and future tool coordinates remain prescribed inputs. Material/force labels and
future target trajectories never enter the predictor.

SoftVT primary contact model is a reduced split model: FEM shape during holding,
rigid pose dynamics with shape frozen at separation. It models gravity, normal
impulses, friction, separation and known static scene supports. Initial public
asset meshes and recorded initial poses define layered heightfields. It does
not model vertical walls, two-way moving-obstacle reactions or elastic impact
deformation. Those limitations must not be hidden behind generic contact claims.
An observed-prefix kinematic test can reject a closed-command grasp when the
tool moves but the object remains stationary. It is not calibrated uncertainty.

scene_fem.py is a separately audited elastic-release diagnostic, not silently
substituted into the primary method. coupled_qualification.py is a separate
synthetic deformable-lattice/rigid-sphere mechanics qualification with equal and
opposite contact forces. Neither qualification establishes benchmark accuracy.

Evidence design
---------------
benchmark-protocol.json freezes 20 SoftVT development episodes and one PhysTwin
development case, the previous 40+3 regression cases, and 40+2 previously
unexamined confirmation cases. Regression means preservation of already examined
results, not a new independent test. Confirmation retains known object categories
and tests new sequences, not unseen-category or unseen-force-family performance.

Comparison includes frozen v11, matched pose reset, selected object frame,
contact with/without scene geometry and detachment detection, geometry-matched
rigid contact, and published/adapted PhysTwin. Extra scene input is disclosed.
Full suffix is primary; hold/release phases are command-defined, not verified
physical attachment labels. All failures, missing jobs and per-sequence changes
are retained. No favorable-only subset is a headline.

Execution
---------
Use runs/phystwin-env/bin/python and requirements.txt. Run jobs.py --create once
to build the locked base/native queue, then jobs.py --worker NAME --kind base or
native with explicit CUDA_VISIBLE_DEVICES for native GPU workers. jobs include
independent subprefix fits and new full-prefix frozen-baseline fits. Do not
recreate a live queue. deadline_guard.py stops registered campaign PIDs only.

Run experiment.py --cohort development before freezing candidate decisions.
method-freeze.json records selected primary rules and source hashes. Then run
experiment.py for regression and untouched_confirmation. report.py --cohorts
development regression untouched_confirmation writes complete lossless scores,
readable tables and plots; incomplete comparisons remain marked incomplete.
render.py --all uses four representative cases selected before test scoring.

Original public assets/data retain their respective licenses and attribution;
asset provenance includes upstream URLs, exact hashes, transforms and scales.
No raw assets/data are included in the repository publication.

Post-audit topology safeguard
----------------------------
The independent mesh audit found that the empirical object-frame residual could
invert inherited tetrahedra. Frozen v12 results remain available, including that
failure. feasibility.py is a separately frozen extension: rigidly align the
inherited feasible FEM shape to the proposed pose, then backtrack the residual
using fixed coefficients until every original element satisfies J>0.03+1e-8.
The 0.03 cutoff comes from the original FEM solver. No future error selects the
coefficient. This guard runs BEFORE contact and velocity initialization.
It guarantees the tested local volume margin, not global collision freedom,
elastic equilibrium or smooth temporal derivatives.

feasibility-protocol.json declares 40 fresh SoftVT episodes (20--29, both suites,
tasks1--2) and PhysTwin single_push_sloth/double_lift_zebra before downloading or
scoring. The original106 cases are secondary post-audit evidence; these42 cases
are the separate fresh confirmation. feasibility-freeze.json records all model
sources after development-only geometric qualification and pipeline tests.

feasibility_jobs.py --create builds the separate124-job queue exactly once.
Use --worker NAME --kind base|native for persistent workers, --repair --watch
for the declared timeout-checkpoint continuation and --status for progress.
The original209-job queue is never replaced. Both queues obey the same absolute
campaign deadline. Native jobs use explicit CUDA_VISIBLE_DEVICES=0,1,or2.

feasible_experiment.py --cohort COHORT --watch produces guarded predictions in
runs/contact-benchmark-v12/feasibility-evaluation, keeping original predictions
unchanged. Supported cohorts: development, regression, untouched_confirmation,
fresh_confirmation. --part/--parts partition work without changing membership.
feasibility_report.py generates the separate complete per-frame diagnostics
archive and paper tables. render_guarded.py --all renders four preregistered
cases. feasibility_verify.py independently recomputes scores and Jacobians via
the original FEM gradients, checks exact jobs/checkpoints and decodes videos.
feasibility_finish.py waits for all declared outputs and runs these final checks.

Campaign services have the prefix contact-benchmark-v12-. Config, queues,
source freezes and progress live on disk; worker execution survives a front-end
session restart. Do not recreate a live queue or change a frozen model source.
Failed experiments and inherited fitting failures must remain in the evidence.

POST-AUDIT MODEL-CLASS SELECTION (all148 exploratory)
model-selection-protocol.json and model-selection-freeze.json preserve the
separate third study. model_selection.py chooses jointly between the original
FEM, a REFITTED FEM without its Gaussian world potential, and nodal rigid
transport. Other force/support terms remain in the field-free model. Validation
uses independent shortened-prefix fits and the fixed adapter shortlist, with
the local-volume guard and identical applicable contact before scoring.
Only a strict greater-than-2-percent validation improvement replaces the exact
saved guarded default. Invalid validation frames invalidate whole candidates;
failed newly selected full predictions remain explicit, never silently replaced.

model_selection_jobs.py --create creates296 separate CPU fits once. Run
--worker NAME and --repair --watch as persistent workers.2400-second job limits
and one1800-second valid-checkpoint continuation are fixed. Terminal failures
write explicit all-NaN candidate arrays and completion records distinguishing
accounted-for work from successful optimization. All cases remain in evaluation.
model_selection_experiment.py --cohort COHORT --watch writes new predictions
without changing either prior study. model_selection_finish.py waits for all
outputs, runs focused tests, generates MODEL-SELECTION-RESULTS.txt and its gzip
archive, then independently checks scores, element volumes, fits and decisions.
render_model_selection.py --all compares recorded geometry, the prior guarded
forecast and the selected model on five predeclared cases. Its watcher service
is contact-v12-model-selection-videos.service (the exception to the usual prefix).
Raw input arrays, fitted parameters and full predictions stay local under runs/.

Evaluator-only amendment: the original PhysTwin manual-marker scorer cannot
construct initial marker correspondences from an unavailable all-NaN family.
model_selection_score_resume.py scores already saved predictions without any
predictor rerun, retaining that manual metric as unavailable and preserving all
failed-frame counts. Finite-initial forecasts use the identical original scorer.
The amendment records source hashes and preserves the21-source model freeze.
Run this score-only resume after the cohort commands if metrics are incomplete.
model_selection_replay.py verifies12 fixed cases by reproducing prefix decisions
and all output arrays; model_selection_verify.py checks all148 results and296fits.
build_paper.py compiles the completed addendum using Tectonic, and copies the
deliverable to output/pdf/contact-benchmark-v12-extension.pdf. Rendered-page
review is recorded separately; the preserved older paper directories are intact.
