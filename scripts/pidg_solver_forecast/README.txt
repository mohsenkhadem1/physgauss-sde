Chronological PIDG solver-driven forecast experiment

Scenes: Balls Reaction, Rubber Duck, Motion Kuro.
No D4RT dependency. Previous full-sequence rendering results remain unchanged.

Why new appearance checkpoints: the older checkpoints observed training frames across the
entire timeline. They cannot initialize a clean future-motion forecast. New prefix-only
PIDG bases preserve the appearance architecture, without importing future observations.

Protocol: original fitting images up to normalized time 0.6; original prefix validation
images for appearance training logs; released training and validation images at 0.6-0.7 for physical
checkpoint/extrapolation selection; released test images after 0.7 for final forecasts.
These datasets were previously inspected, so this is a new chronological split, not a
pristine blind benchmark. Raw images/checkpoints/trajectories remain local.

Three GPU workers: train_prefix.py --scene SCENE --gpu 0|1|2. Each runs fitting-only flow
preprocessing, a 50k native base, then separate 5k native and dense-hybrid refinements.
Hybrid weight 1e-5 is fixed from earlier work. GPU 3 is untouched.
finish_scene.py waits for training, builds geometry, fits mechanics, seals and evaluates.
report.py generates all-case summaries and figures; audit.py verifies split/checkpoint
integrity. Geometry and model unit tests can run before training finishes.

Solver architecture: high-opacity prefix Gaussians define occupied voxels, with hole filling
and disconnected components preserved. Dense Gaussians bind through affine-exact tetrahedral
coordinates. Initial covariance reproduces the author's rasterizer convention. A frozen
learned material-energy family, one material scale, damping, 12 spatial Gaussian interaction
kernels with 8 temporal load knots, positive attachment stiffness and a global body load
drive symplectic tetrahedral integration. Scene units are normalized, not calibrated SI.
All appearance motion in solver forecasts comes from these integrated mechanical states.

Observed prefix motion initializes interior states by weighted projection of the native
prefix model; orientation-preserving shrinkage is recorded. Training uses 300 short RGB
multiple-shooting updates from these reconstructed segment states, not unknown true nodes.
Initialization compares derivative-force multipliers 0, 0.1, 1 on validation. The physical
checkpoint and one of four predeclared force extrapolations (constant, exponential decay,
zero, last linear slope) are chosen using only the forecast-validation images. Final test
rollouts freeze parameters and integrate from the last observed state with no future target
or force access. This is iterative training and forward-only prediction.

Limitations: the occupied volume is a proxy rather than a known solid reconstruction;
monocular depth is relative and used in native fitting only where the original recipe does.
No true mass, forces, constitutive parameters or 3D future states are provided. This public
adapter does not yet enforce explicit inter-object contact; unknown interactions are fitted
Gaussian fields. No metric SI inference or calibrated uncertainty is claimed. Novel actions
and novel camera poses without captured targets are not quantitative benchmarks.

Metrics: RGB/foreground/union PSNR, SSIM, LPIPS-Alex, silhouette centroid/IoU, and TV-L1
flow disagreement on consecutive held frames. The last metrics are image-motion proxies,
not true 3D errors. Positive Jacobians and internal-force balance are numerical diagnostics,
not certificates of physical truth. All rollout failures remain visible. Constant-velocity
Gaussian extrapolation provides a simple additional control.

Implementation updates before any test scoring: static appearance refinement uses 2000 updates;
121 observed prefix trajectory samples initialize mechanical states. Forecast validation also
selects projected versus mechanically integrated-tail terminal state. Native PIDG selects its
better base/refined checkpoint on that same validation interval. Sampled interbody penetration
is reported when multiple reconstructed components exist; it is not enforced by this adapter.

Final implementation details: Balls uses six instances specified from the first fitting
image and spatial/colour clustering, rather than claiming fully automatic segmentation.
After initial projection, local orientation-constrained nodal refinement preserves J>0.2.
Static appearance fitting chooses world/material SH directions on prefix validation.
A failed raw-velocity Duck attempt is retained. Target-free timestep tests isolated noisy
nonrigid initial velocities. Per-component rigid twist plus residual factors 0, .25, 1
are selected on validation; the dynamics themselves remain deformable. Initialization
also selects material scales .03, .3, 1. Constant velocity independently selects filtering.

Completed result: all 41 final frames per method rendered. The target was not met.
Solver full-image PSNR, LPIPS and flow agreement are worse than selected native PIDG
on all three scenes. Balls foreground PSNR improves, but does not beat constant velocity.
Prefix appearance already loses 10.20 / 14.19 / 4.91 dB on Balls / Duck / Kuro after
binding/refinement. Fixing that representation bottleneck precedes further forecast claims.
See reports/pidg-solver-forecast-v01/REPORT.txt and output/pidg-solver-forecast-v01/.
