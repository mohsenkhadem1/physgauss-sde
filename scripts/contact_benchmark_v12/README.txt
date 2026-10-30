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
