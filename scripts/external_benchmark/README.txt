External position-driven deformation benchmark (v11)
===================================================

Two external datasets: SoftVTBench and PhysTwin. This directory adapts the
material-reference/world-boundary FEM idea to arbitrary initial geometry.
It is a per-sequence physical identification and future prediction experiment,
not an RGB reconstruction pipeline or a manipulation-policy leaderboard entry.

Inputs and separation
---------------------
Known: initial geometry, observed-prefix point positions/visibility, time-indexed
future tool positions, gripper opening/release commands when supplied by the data.
Unknown: material coefficients, attachment compliance, effective gravity/material
ratio, world-potential coefficient. True forces and material labels are unused.
Only the evaluator reads future object shape/manual markers. data.py loads those
into a separate evaluation dictionary, removed before construction of the FEM.

SoftVTBench initial volumetric nodes and 48 upper-node observations are privileged
synthetic geometry, equally available to fitted methods. The camera perception
problem is outside this experiment. Tool poses are transformed from robot-local
to world coordinates. The gripper/object transform is calibrated from those same
sparse observations at an input-defined grasp within the prefix. Release follows
the binary command, because recorded gripper width may remain closed on release.

PhysTwin uses the authors' processed tracks, initial surface/interior geometry,
controller tracks and original splits. 300 initial nodes define the mechanics;
all visible observed tracks remain available in the fitting loss. Initial-geometry
probes remove Delaunay tetrahedra bridging empty spaces between limbs. This is
an approximate volume mesh, not a recovered manufactured rest configuration.

Physics
-------
Neo-Hookean tetrahedral energy in material coordinates, fixed Poisson ratio .35,
with spatial stiffness variation (one feature for SoftVTBench, six Gaussian
features for PhysTwin). World terms are gravity, a unilateral initial floor,
a localized compliant support, and a scalar Gaussian potential. Position-driven
attachments supply the interaction. The no-world-potential ablation retains
ordinary gravity, contact and attachments; it removes only the additional learned
Gaussian potential, not all boundary conditions.

Analytic energy/residual/tangent with Newton equilibrium solves. L-BFGS-B fits
physical ratios from the observed prefix using implicit differentiation. Before
release predictions are quasistatic. After SoftVTBench release, five implicit-Euler
substeps per frame use mass/stiffness implied by the learned gravity ratio.
Initial observed shape is treated as stress-free. Ratios are point estimates;
these runs do not provide calibrated material or boundary posterior intervals.
New basket/contact geometry is not provided to any predictor. This matters on
release, and full suffix and input-defined held-grasp/release scores are separate.

Baselines
---------
Persistence; position-controlled rigid transport with ballistic release and the
same initial floor; position-feature ridge regression; no-world-potential FEM;
released author PhysTwin predictions on PhysTwin; and an adapted native PhysTwin
spring-mass optimizer on SoftVTBench using the same sparse prefix and initial
geometry. The adapted baseline uses native physical kernels and losses, learned
springs and native floor collisions. Its controller springs are explicitly removed
on release. It is not an official author SoftVTBench result. Author PhysTwin
predictions use author preprocessing and initialization. Available observation
prefixes match, but optimization schedules and representations differ.

Reproduction
------------
1. From repository root, run bash scripts/phystwin/setup.sh to create the existing
   pinned PhysTwin environment and headless imports. Install scripts/external_benchmark/requirements.txt in that
   environment. Requirements are in scripts/phystwin/requirements.txt; recorded
   environment: Python 3.10, torch 2.4.0+cu121, numpy 1.26.4, scipy 1.15.3,
   warp-lang 1.6.2. Original PhysTwin revision:
   81c718790a37e5e0102eb77af2c6edd34a9db25f.
2. runs/phystwin-env/bin/python scripts/external_benchmark/acquire.py
   Fetch each of single_lift_sloth, double_stretch_sloth, single_lift_dinosor,
   double_stretch_zebra with scripts/phystwin/fetch_example.py --case CASE.
   Raw files stay under runs/ and are not committed.
3. Read reports/external-benchmark-v11/frozen-protocol.json for exact primary
   command vectors, cohort, inputs and SHA256 hashes. Execute commands from the
   repository root. The original 12-hour campaign deadline is enforced by
   common.check_deadline. A later reproduction must explicitly choose a new
   deadline in configs/external_benchmark_v11.json and use a fresh output path or
   archive prior runs; do not silently overwrite original evidence.
4. For the original campaign, campaign.py --worker NAME shares a locked job queue.
   Workers use CUDA_VISIBLE_DEVICES=0,1,2 respectively. Each child saves provenance,
   fitted parameters, predictions and numerical diagnostics. Runs exceeding 30
   minutes are retained as failed jobs. deadline_guard.py only stops registered
   campaign processes with matching PID start times.
5. After all jobs finish, run scripts/external_benchmark/aggregate.py. It verifies
   frozen prediction/scoring source hashes, scores all 43 selected sequences and
   keeps failures and missing jobs explicit. The complete JSON is published as
   results.json.gz; results.json is the local uncompressed mirror. CPU tests:
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 runs/phystwin-env/bin/python -m pytest
   tests/test_external_fem.py -q
6. render.py makes shared-scale actual/predicted point-cloud videos. These are
   geometry playbacks, not raw camera videos or photorealistic Gaussian renders.

Interpretation
--------------
Held-out suffixes test new positional trajectories after per-sequence fitting.
They do not establish zero-shot transfer to unseen objects/materials or a withheld
force-family benchmark. SoftVTBench's official DSR manipulation metric is not
used. Scores average frames within each sequence, then sequences equally; frames
are not independent statistical samples. Numerical failures are reported next to
errors conditional on finite predictions. Do not hide incomplete sequences.

Sources
-------
https://softvtbench.github.io/
https://github.com/TuojingAI/SoftVTBench
https://huggingface.co/datasets/Arthur12137/SoftVTBench
https://github.com/Jianghanxiao/PhysTwin
https://huggingface.co/datasets/Jianghanxiao/PhysTwin
Data and upstream code retain their original licenses and attribution.

Release correction and reporting variants
----------------------------------------
The physical_release_ results remove the static rest-position gauge spring when
inertia is active. This prevents an artificial restoring force after release,
without changing fitted parameters or physical energy terms. All 40 episodes
and both FEM variants receive this correction. The primary and adaptive-only
variants remain in the results archive. dynamic_fem.py implements the correction;
the translated-body free-fall test verifies it. General release temporal
convergence remains unestablished. The positive-volume solver and localized
gripper adapter are development diagnostics, not promoted production models.
