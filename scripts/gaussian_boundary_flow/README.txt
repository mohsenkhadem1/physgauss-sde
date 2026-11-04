GAUSSIAN BOUNDARY FIELDS + CONDITIONAL FLOW MATCHING
Development campaign, 4 November 2026

Implemented
- Invisible anisotropic Gaussian surface gates, distinct from rendering Gaussians.
- Reference-area quadrature for tractions and Robin stiffness/damping operators.
- Task-frame coordinate transforms; PSD stiffness/damping parameterization.
- Exact prescribed DOFs and reaction forces through elimination.
- Explicit temporal target programs, activation/release and finite command horizons.
- Isolated plane-impact/friction primitive, tested separately (not a coupled rollout).
- Existing tetrahedral elastic law linearized at rest; implicit dynamic integration.
- Conditional rectified flow over six uncertain physical parameters: Young modulus,
  support stiffness and x-position, applied force magnitude and x-position, damping.
- Train on simulator examples; sample a physical explanation once; integrate future
  motion under new commands. No per-frame inverse fitting or shooting.
- Separate generative flow time s from physical time t. No physical SDE is needed.
- Existing Gaussian renderer for synthetic RGB inference/interpolation/novel views.
- Local observability, additional-view acquisition and predictive calibration tests.

Architecture
Observed prefix -> context encoder -> conditional flow -> physical samples
               -> Gaussian boundary operators + elastic integrator -> Gaussian render
For track experiments the context is nine frames of eight 2D projected corner tracks.
The RGB prototype uses nine 24x24 RGB frames and a train-only 64-dimensional PCA
encoder. Camera, initial geometry/state, density and appearance are known. Force
direction is fixed; the future command is supplied. Discrete boundary type is not
inferred. The six continuously varying parameters above are hidden from inference.

Experiments and retained evidence
1. runs/gaussian-boundary-flow-v01: 8-node controlled track prototype, 8192 training
   worlds, 128 test worlds. Positive inference result; NOT a physically qualified
   reference. Its large spatial error and up to 32% strain motivated a new experiment.
2. runs/gaussian-boundary-flow-rgb-v01: same coarse prototype, raw synthetic RGB
   input. Future RMSE 0.962-0.992 mm versus ridge 3.003-3.420 mm. Novel-camera
   rendering improves under this fixed-appearance synthetic renderer. Intermediate
   withheld frames: 53.81 dB versus 49.11 dB simple RGB blending. No specialist
   video interpolation baseline or real RGB reconstruction claim.
3. runs/gaussian-boundary-flow-refined-v03: 64 physical nodes, 8192 independent
   training examples, a new 128-world test seed, smaller-strain range, dt=.0025 s.
   Flow future errors 0.189-0.206 mm; same-bank neural regression 0.338-0.358 mm;
   ridge 1.457-1.507 mm. Four future force programs: continued sine, load release,
   triangular loading and reversal. These share the exact observed prefix.
   The neural baseline uses the same 15000 optimization steps and 256 batch size;
   its architecture and objective differ. Flow uses 32 posterior samples per case.
4. Cross-mesh test: 64 fresh worlds generated on 125 nodes, inferred/predicted on
   64 nodes; force reversal. Flow .250 mm, ridge 1.040 mm, true-parameter coarse
   simulation .111 mm. Raw nominal 90% coordinate coverage is only 80.2%.
5. New-view diagnostic: 32 independent worlds, 128 samples, one additional view
   of the last observed time with 0.5 mm noise. Error .205 -> .093 mm. Selected
   and fixed policies tie; random .095 mm. This is approximate proposal reweighting,
   with ESS recorded, not a validated exact posterior or superior active policy.
6. Predictive calibration: 64 separate finer-domain calibration worlds and 64 fresh
   test worlds. A split-conformal additive trajectory envelope achieves 93.75%
   whole-trajectory coverage. Average width .810 -> 1.403 mm. This requires
   finer-domain calibration data; it does not calibrate material/force posteriors.
7. position-action.json: unseen smooth 6 mm upward motion of the inferred attachment,
   applied load removed after .4 s. This tests a supplied POSITION command, with
   no force refitting. Read its measured values in the consolidated report.

Numerical qualification
- Temporal refinement .0025 -> .00125 s changes trajectories by .0799 mm (16 cases).
- CPU/GPU maximum displacement difference 1.77e-7 m (four cases).
- Independent eight-case spatial check: 27->64 nodes .860 mm; 64->125 nodes .336 mm.
  The latter is still larger than some same-mesh inference errors. Cross-mesh
  evaluation is therefore reported explicitly; do not call 64 nodes converged.
- Maximum absolute strain in that refined eight-case check stays below 10%.
- Linear elasticity is restricted to small strain; large rotations, drops and
  coupled contact require the separate nonlinear/contact continuation.

Assets
reports/gaussian-boundary-flow-v01/results.json: consolidated measured evidence.
reports/gaussian-boundary-flow-v01/audit.json: scope and preservation checks.
reports/gaussian-boundary-flow-v01/results.png and results.pdf: quantitative figure.
reports/gaussian-boundary-flow-v01/refined-actual-vs-predicted.mp4: median and worst
force-reversal cases, orbiting camera, true deformation scale, 0.5x playback.
Its surfaces interpolate output corner nodes; it is a geometry visualization,
not a photorealistic renderer result. Selection/provenance are recorded beside it.
Checkpoints, raw synthetic image banks and videos stay local.

Reproduction
From repository root, with the existing virtual environment:
  .venv/bin/python -m pytest tests/test_gaussian_boundary_flow.py tests/test_physics.py tests/test_forces.py -q
  .venv/bin/python -m scripts.gaussian_boundary_flow.campaign --out runs/NEW-PROTOTYPE
  .venv/bin/python -m scripts.gaussian_boundary_flow.rgb_campaign --out runs/NEW-RGB
  .venv/bin/python -m scripts.gaussian_boundary_flow.refined_campaign --out runs/NEW-REFINED --train 8192 --steps 15000 --test 128 --seed 39019 --test-seed 77183
The post-training GPU evaluators currently explicitly read the preserved v03 run:
  .venv/bin/python -m scripts.gaussian_boundary_flow.evaluate_refined --device cuda:0 --kind sine
  .venv/bin/python -m scripts.gaussian_boundary_flow.evaluate_refined --device cuda:1 --kind release
  .venv/bin/python -m scripts.gaussian_boundary_flow.evaluate_refined --device cuda:2 --kind reverse
  .venv/bin/python -m scripts.gaussian_boundary_flow.evaluate_refined --device cuda:1 --kind triangle
  .venv/bin/python -m scripts.gaussian_boundary_flow.point_baseline
  .venv/bin/python -m scripts.gaussian_boundary_flow.cross_mesh
  .venv/bin/python -m scripts.gaussian_boundary_flow.acquisition
  .venv/bin/python -m scripts.gaussian_boundary_flow.calibration
  .venv/bin/python -m scripts.gaussian_boundary_flow.position_action
  .venv/bin/python -m scripts.gaussian_boundary_flow.report
Only GPUs 0-2 were used. The sandbox hides NVIDIA devices; authorized execution
outside it worked. GPU 3 was not used. Training checkpoint v03 is complete; its
original CPU evaluation was interrupted deliberately and replaced by GPU workers.
The interrupted v02 directory contains only its abandoned plan, not results.

Remaining campaign work
- Transfer this inference model to the existing real PhysTwin/SoftVT benchmarks;
  their prior scores are preserved and must not be reassigned to this new model.
- Unknown discrete boundary types and multiple interacting kernels; richer meshes.
- Coupled unilateral contact, release/drop/impact and nonlinear finite rotations.
- Unknown appearance/geometry/cameras; RGB training for the finer physical model.
- Test new actions selected by uncertainty against strong fixed/random policies.

Related foundations
Task-parameterized impedance GMMs: https://calinon.ch/papers/Calinon-ICRA2014.pdf
Flow matching: https://arxiv.org/abs/2210.02747
Neural Gaussian Force Fields: https://arxiv.org/abs/2602.00148
Gaussian force fields alone are not a novelty claim. The research direction is
boundary identification, observable ambiguity, and improvement from new evidence.
