PIDG / frozen material-law integration experiment
================================================

This is a NEW integration ablation. It uses the existing learned, frozen
ScaleSeparatedMaterialLaw inside official-source PIDG reconstruction. It is not
an evaluation of the complete position-driven forward solver, contact model,
force inference, or uncertainty model from the main paper.

Data and fitting
----------------
Public PIDG balls_reaction: 150 RGBA frames and supplied camera poses.
101 fitting frames, 11 validation frames selected from the released training
split, 38 untouched released test frames. A single frame-index clock replaces
the release's independently normalized train/test clocks. Raw data stay in runs/.
TV-L1 flow is recomputed between actual fitting-frame neighbors; no held-out
image participates in optical-flow preprocessing. Released per-image depth is
used only for fitting frames. The released initial point cloud is excluded
because its observation provenance is unknown. All methods start from the same
seeded 100,000 random points.

Architecture transfer
---------------------
PIDG predicts a canonical-to-current map phi_t(X)=R_t(X) X+d_t(X).
We evaluate F=d phi_t/d X by centered finite differences at 256 deterministic
canonical Gaussian centers (epsilon=0.01 in dataset coordinate units).

The added loss is
  lambda * mean[ W_theta(F, z)/12000 + 10 max(0, 0.1-det(F))^2 ],
with the existing frozen shared material checkpoint and z=(0,0,-1).
The learned energy is evaluated only where det(F)>0.05. Elsewhere it receives
identity, and the explicit inversion term supplies a penalty. This is a
piecewise extension for optimization; it is not a validated constitutive law
for inverted elements. lambda is selected from {0.001,0.01} on validation only.
Rigid rotations/translations produce essentially zero energy, unlike a raw
position or displacement penalty. Frozen material weights receive no updates.

The added term is a material regularizer, NOT a solved momentum balance. It
provides neither physical units nor a recovered material estimate on these
images. The unchanged PIDG optical-flow and momentum terms remain enabled in
all main runs. Known upstream residual/rasterizer scientific issues documented
in the previous source audit are preserved equally in both arms.

Execution
---------
Use the isolated official environment:
  /home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env/bin/python

prepare.py downloads data and defines the split. preprocess.py requires the
existing runs/deform360-env environment's OpenCV contrib TV-L1 implementation.
run.py runs official training with explicitly recorded compatibility edits,
optionally adding the frozen energy. Test evaluation is prohibited until
selection.json records the validation choice. finish.py monitors completion,
evaluates validation, seals selection, evaluates test, and builds figures.

Each main run uses 50,000 iterations at 320x180, seed 0. GPUs 0,1,2 are used;
GPU3 is untouched. These are controlled reproduction runs, not the authors'
reported leaderboard scores. All preprocessing/compatibility differences are
listed in reports/pidg-comparison-v01/protocol.json and per-run run.json files.

Metrics and visual policy
-------------------------
Standard RGB PSNR, SSIM, correctly normalized LPIPS-AlexNet, foreground PSNR
and RMSE. The separate channel-averaged PSNR is retained to reproduce the
upstream training logger; it is not substituted for standard RGB PSNR.
Sampled det(F)<=0 and absolute volume changes are geometry sanity diagnostics,
not physical accuracy: no true material, forces, volume histories or physical
time units have been supplied to this adapter.

Selected favorable images must be explicitly labelled as such; every test
frame remains in the aggregate and per-frame report. Ground truth, baseline,
and hybrid use identical images, cameras, crop coordinates, and display range.
The least favorable example is saved separately. No artificial sharpening,
image generation, deformation exaggeration, or baseline degradation is used.

Recorded outcome and refinement
-------------------------------
The initial 50k experiment is preserved as v01. Based on validation only, v02
adds 5,000 fitting steps to PIDG and two copies of the validation-selected
hybrid, relaxing its material weight to 1e-5 or 1e-4. Each arm restarts its
optimizer and velocity/stress network because upstream checkpoints omit their
states. Gaussian densification is disabled. The selected final weight is 1e-5.
This is an explicitly recorded warm restart, not an exact optimizer resume.

On the 38 test frames at 320px, v02 PIDG obtains 49.54 dB RGB PSNR and the
material-prior integration 48.16 dB. The hybrid wins whole-image PSNR on 6/38
frames and foreground PSNR on 5/38. At 640px rendering, overall PSNR is nearly
tied (34.48 vs 34.41 dB); hybrid foreground RMSE is about 1.3% lower. Sampled
inversion fractions are 0.001336 vs 0.0. These diagnostics are not calibrated
physical accuracy. The two selected 640px crops have about 29% and 26% lower
RGB RMSE with the hybrid; these are local examples, not aggregate superiority.
Full and unfavorable results remain in reports/pidg-comparison-v01 and v02.

The 4,391-byte previously learned material checkpoint is included unchanged
at reports/pidg-comparison-v01/shared-material-law.pt. Raw datasets, rendered
frame arrays, and full per-scene reconstruction checkpoints remain local.
