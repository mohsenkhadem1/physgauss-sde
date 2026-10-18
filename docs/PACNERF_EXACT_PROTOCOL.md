# Corrected PAC-NeRF transition-integral comparison

The historical four-point-integral experiment is superseded. This correction
uses exactly the same prepared images, initial geometry, frozen modes/colors,
transferred neural network, noise scales, mechanical/load priors, view schedules,
renderer and frame-held ground reaction. Only the **linear-system covariance
and constant-load integrals** are replaced by balanced augmented matrix
exponentials. It does not make nonlinear contact integration exact or reproduce
the original MPM simulator. Known rendering/contact model limitations remain.

The numerical source fingerprint is
`eb703c41c125e971edae446bcad13d8f80a0ca5a34efb9f6a7b156762bf9f655`.
Old references/chains cannot be resumed into this target. The old source is
preserved at commit 2f07671 and in the local `physgauss-legacy-audit` worktree;
old raw data remain in `runs/pacnerf-production`.

Each mode is balanced before computing its 3x3 affine input integral. Every
mode pair uses a stable 5x5 augmented Kronecker-sum exponential to integrate
the full, possibly correlated acceleration spectrum. This avoids a fixed
quadrature's high-frequency aliasing and the anti-damped block in a Van Loan
calculation. Independent analytic/full-matrix/derivative tests and the 256
historical state probes validate the correction; they are not new posteriors.

The four arms remain constant/neural edge-noise amplitude crossed with
analytically marginalized/explicit joint Gaussian loads. Force priors and
integrated physical-path posteriors match within each noise choice. Constant
amplitude still has geometry-dependent covariance. No network is trained on
these test trajectories. Supplied initial geometry and eleven-camera initial
appearance bootstrap remain explicit conditions; 4/6 labels mean 3/5 later
dynamic images over fourteen timestamps, not unknown-shape monocular recovery.

Inference retains the historical physical path coordinates, exact nonlinear
image/physics gradients, stationary observed-Hessian Gaussian reference and
Metropolis-corrected split HMC. The observed Hessian is evaluated in bounded
reverse-mode batches to control memory, and image-coordinate scales use their
actual stored values. This changes neither the objective nor the intended
Hessian. Mode stationarity requires decrement <.02 and positive observed
curvature. All substantive settings and serialized container types are checked
before resuming.

The full plan has sixteen cases: Torus/Bird, four/six selected images, both
noise types and both force treatments. Four chains discard 500 warmup
transitions and retain at least 2000 draws each. Original fixed-kernel gates
at 2000/4000/8000 are retained: all state coordinates, conditional force means,
log target and covariance normalizers require Rhat <1.01 and bulk/tail ESS >=400,
with no frozen chains. Passing shorter checkpoints cannot stop a full run.
Warmup is recoverable every 25 transitions and retained states every <=50
draws. Source, input hashes, reference, kernel, RNG stream and prefixes must
remain consistent across any continuation. Gradient counters exclude unsaved
work after the last checkpoint; report all attempt times rather than presenting
a restart's seconds as total runtime.

Before the full launch, an observed-data-only Bird four-view development pilot
tests all four arms (8/9/10/11) with 100 warmup/100 retained draws and a 25-minute
compute cap per arm. It computes no future-geometry reconstruction or coverage
score and is never eligible for ranking. No pilot chain is reused as a full
study's posterior. The full comparison uses fresh directories and seed
140001 plus case index, as before. Runtime/library versions remain pinned to
Python 3.12.4, Torch 2.14.1 CPU and NumPy 2.0.0, with two Torch threads and
greedy opt-einsum contractions. `pacnerf-exact.yml` reuses the byte-identical
original prepared inputs and records source/worker hashes in each case.

Full reconstruction scoring only follows inference. Original component
coverage/width, gap coordinate RMSE, held-out full/region RGB scores, independent
raw replay, paired marginal/joint Monte Carlo screens and force total covariance
E[S]+Cov[m] remain required. Incomplete/unconverged cells are identified and
excluded from quality rankings. Two deterministic objects do not establish
statistical calibration or force accuracy; no force ground truth is supplied.
