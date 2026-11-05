PIDG Balls Reaction: paper mechanics connected to Gaussian rendering
===================================================================
This pipeline executes the paper's NeuralFEM arbitrary-mesh equilibrium
adapter, not PIDG with an added material regularizer. It uses the existing
frozen ScaleSeparatedMaterialLaw, inferred Gaussian attachment footprints,
implicit equilibrium, tetrahedral Gaussian binding, covariance transport,
and the official PIDG rasterizer. PIDG's DeformModel is not queried at render
or mechanics-replay time.

Scope and inputs
----------------
The public release has RGB and camera poses, not a mechanical scene. The
adapter assumes six separately segmented, approximately spherical objects.
A training-only PIDG reconstruction supplies an initial radius/depth gauge.
Hue-connected training-image regions provide position/ellipse observations.
Each object has 13 FEM nodes, 20 tetrahedra and six Gaussian attachment patches.
These patches are inferred boundary hypotheses, not observed actual grippers.
Gravity is negligible in this dimensionless unidentified-scale adapter.
There is no coupled collision solve or calibrated material/force posterior.
This is the paper's equilibrium adapter, not the Q2 transient contact adapter.

The first versions preserve negative results: v01 reconstruction-affine targets;
v02 an inadequate color grouping; v03 corrected connected-region tracking with
SH degree 1; v04 the same mechanical inputs with SH degree 3. All selection is
on the 11 validation images. 101 images fit the model; 38 test images are read
only after the final selection file is written. Initial radius estimates depend
on a PIDG checkpoint trained on those same 101 fitting images. This is a
reconstruction-assisted pipeline, not a claim of independent initialization.

Actual fitting loop
-------------------
1. Fit shared attachment strength per body using NeuralFEM on training-derived
   position observations; the material network stays frozen.
2. Replay inferred position controls with NeuralFEM; attach 768 Gaussians/body.
3. Fit SH appearance, opacity and bounded material-coordinate appearance offsets.
4. Jointly optimize a 12-knot boundary-target correction, attachment/material
   descriptors and appearance from training RGB. Every mechanics update uses
   a converged NeuralFEM solve and its implicit adjoint, not a deformation MLP.
5. Select on validation foreground PSNR, freeze, then score released test RGB
   and render the predeclared off-path camera sweep.

Both nodes and Gaussian covariance follow mechanics. Camera directions are
pulled into the material rotation frame for SH radiance. Six independent meshes
prevent material mixing across objects. Binding includes explicit affine
extrapolation just outside the coarse tetrahedral surface. Appearance offsets
are bounded to 25 percent of initial radius; covariance scale is bounded.
These are disclosed appearance freedoms and geometry priors, not proof of
physical accuracy. Inferred full-sequence controls imply interpolation, not
causal forecasting or new-action response recovered from video alone.

Reproduction
------------
Use the repository .venv for fit_mechanics.py and cache_mechanics.py; use the
isolated pidg-official-v01/env for CUDA rendering and joint.py. All commands
run from repository root. PIDG_MECHANICS_ROOT chooses a run directory;
PIDG_SH_DEGREE selects 1 or 3; PIDG_GPU selects physical GPU 0, 1 or 2 for
rendering. fit/cache use GPU 1. Raw data/checkpoints remain under runs/.

prepare.py produces the v01 reconstruction initialization. refine_inputs.py
produces v03 corrected training geometry. Run fit_mechanics.py, then
cache_mechanics.py; for v04 reuse that sealed geometry and mechanics state,
set PIDG_SH_DEGREE=3, then train_appearance.py (2000 updates), joint.py
(80 updates), and joint.py with PIDG_JOINT_RESUME=1, PIDG_JOINT_STEPS=400,
PIDG_JOINT_EVERY=100. The second joint phase is a declared warm restart with
fresh optimizer state. Preserve the first phase's reports/checkpoint.

seal_selection.py must precede evaluate.py. report.py and video.py assemble
all-case metrics and fixed-camera figures; no selected crop is used as an
aggregate superiority claim. The camera sweep uses first/middle/last test
times and yaw offsets -90,-60,-30,0,30,60,90 degrees at 640x360.

Verification
------------
Run test_bridge.py, test_tetra.py and test_implicit_boundary.py in
scripts/mechanics_gaussian_bridge. They cover affine exactness, covariance
transport, derivative parity, orientation rejection and finite-difference
agreement of the boundary adjoint. Input/export seals and evaluation results
record the fitting/test separation and limits of each claim.
