# Analytically marginalized forces (version 0.5)

## Scientific purpose

Test whether eliminating a Gaussian force block improves numerical inference
relative to alternating mechanics/force updates and joint explicit-force
optimization, using exactly the same sparse images and probabilistic model.
Elimination alone does not create new information or resolve stiffness/load
ambiguity. When both solvers succeed, their matched answers should agree.

## A deliberate model change

The v0.4 simulator has 15 smooth force coordinates, optionally augmented by
three internal Brownian coordinates per step. Its distribution over arbitrary
free-node trajectories is singular: there are 12 free-node velocity coordinates
per step. A generic inverse of its trajectory covariance does not exist.

The new trajectory model therefore adds **full-rank independent free-node
velocity diffusion**, with fixed amplitude 0.02 (m/s)/sqrt(s), equivalently
0.02 m s^(-3/2). This is a declared
nodal discrepancy/external impulse model, not balanced internal noise, a
thermal law, or numerical jitter hidden in a matrix factorization. Each
innovation has velocity covariance sigma_v^2 dt I. In momentum units its
standard deviation is m_i sigma_v sqrt(dt), in Ns. It is not a claim that
these are the true object disturbances. The current version does not combine
this floor with the old state-dependent neural internal SDE.

All three NEW solvers use this exact same model. Historical v0.4 comparisons
have identical images, cameras, geometry and smooth force prior but different
process noise and posterior approximation. Do not attribute differences against
the old simulator solely to marginalization. Density MAPs in trajectory and
white-noise coordinates are also different estimation conventions; MAP is not
invariant to arbitrary changes of coordinates.

## Trajectory coordinates and physical equations

Unknowns z comprise eight standardized scene/material coordinates theta and
the free-node velocities v_1,...,v_S at EVERY solver step. Velocities are stored
in coordinates scaled by a fixed 0.1 m/s; this is numerical conditioning, not
an extra prior or adjustable uncertainty temperature. Initial displacement
and velocity have the existing priors. Pinned nodes remain exactly fixed.

Positions obey semi-implicit Euler exactly:

    x_{k+1} = x_k + dt_k v_{k+1}.

With the existing Neo-Hookean elasticity, gravity and mass-proportional drag,
define the mass-normalized momentum residual:

    r_k(z) = v_{k+1} - v_k
             - dt_k [f_elastic,theta(x_k)/m + g - drag_theta v_k].

Keep only free-node components. No noisy image acceleration estimates are
formed. Deformation validity is checked at every solver state including the
last state. Known initial shape, density, supports, appearance and cameras
remain strong assumptions.

The finite Matérn-3/2 inducing force prior is unchanged: five temporal inducing
points, three directions, 15 independent whitened coefficients a ~ N(0,I),
0.05 N total-force prior scale and the fixed mass-weighted spatial profile.
On this cube the profile gives uniform free-node acceleration; this does not
infer force location. Let F_k(a) be its nodal force in N. Then

    r(z) = G a + eta,        eta ~ N(0,Q),
    G_k a = dt_k F_k(a)/m,
    Q = diag(sigma_v^2 dt_k), repeated over free-node coordinates.

G and Q are fixed in this experiment. The causal map from velocities to
residuals is block lower triangular with identity diagonal. In stored scaled
velocity coordinates its determinant is scale^(12 S); the objective includes
this constant Jacobian. This is a normalized causal trajectory model, not an
arbitrarily weighted PDE residual. Initial-state prior constants and image
normalization constants fixed across all fits can be omitted.

## Exact elimination

Integrating all 15 force coefficients yields

    r(z) ~ N(0,C),           C = Q + G G^T.

The nonlinear negative log posterior is

    J(z) = 1/2 ||R_obs(X(z)) - y||^2_{Sigma_y^-1}
           + 1/2 ||theta||^2
           + 1/2 r(z)^T C^-1 r(z) + 1/2 log|C|
           + S*12/2 log(2 pi) - S*12 log(velocity_scale).

Observed RGB only enters fitting. Missing-time and novel-view targets enter
evaluation only. Nonlinear mechanics and rendering remain nonlinear; this is
not a closed-form full inverse-rendering solution.

For efficiency, set W = Q^-1/2 G and A = I + W^T W. The implementation factors
only the 15-by-15 A, uses the determinant lemma for log|C|, and a thin SVD of W
for a stable symmetric inverse square root. It does not invert a dense C or
subtract two large quadratic forms. The determinant is retained even though
it is constant for the fixed hyperparameters in this study. Hyperparameter
learning would have to differentiate it and retain all changing normalizers.

## Recovering a force distribution

At any trajectory/material state:

    a | z,y ~ N(mu(z), A^-1),
    mu(z) = A^-1 G^T Q^-1 r(z).

Images do not add another force likelihood after conditioning on the complete
trajectory. Over a trajectory posterior approximation:

    E[a|y] = E_z[mu(z)],
    Cov[a|y] = A^-1 + Cov_z[mu(z)].

Both terms are saved. Reporting A^-1 alone would omit uncertainty about motion
and mechanics. Local trajectory/material draws are paired with conditional
force draws, retaining their dependence. The force mean and full coefficient
covariance use this Rao-Blackwell decomposition; percentile bands and CRPS use
finite conditional draws. Finite-sample bands and analytic-moment estimates
need not agree exactly. Existing `evaluate_force_posterior` queries the saved
coherent force moments/functions within the recording interval.

## Matched numerical comparison

- **Alternating:** solve the force conditional exactly, then take a mechanics/
  trajectory Gauss-Newton step with force held fixed and a decreasing line
  search. Repeat.
- **Joint:** optimize trajectory/material and explicit force coefficients in
  one joint Gauss-Newton system with a decreasing line search.
- **Marginalized:** optimize only trajectory/material with the integrated
  force objective and its coupled Gauss-Newton system.

The force conditional covariance is constant here. Thus the marginalized
objective and the joint objective profiled at its exact conditional force
mean differ by a constant. Their trajectory MAP stationary points coincide.
Eliminating the force block from a joint local Gaussian uses the Schur
complement and gives exactly the marginalized Gauss-Newton state precision.
These are tested numerical identities, not a prediction that marginalization
must give lower reconstruction error. Alternating steps omit this coupling
while taking the mechanics step, so they can converge more slowly.

Each method starts from the SAME observation-only v0.4 force-only MAP fit,
lifted to solver-step velocities with zero new discrepancy. Initialization
cost is reported separately and must be included when considering end-to-end
cost. The default plan uses 20 outer iterations per method, not equal CPU
budgets. Full settings: 21 frames, six irregular observed 24x24 RGB frames,
60 solver steps, four truth conditions, seed 4101, 128 local posterior draws
and 128 separate independent importance-audit proposals. There are 728 state
coordinates versus 743 joint coordinates; the historical force-only simulator
has just 23 coordinates. Exposing trajectories increases dimension substantially.

Image derivatives use forward-mode AD with respect to the 12 free-node
positions in each observed frame, then chain the exact linear position map.
Mechanics Jacobians use AD. No finite-difference image perturbation crosses
depth-sort branches. The renderer itself still has hard depth sorting, so
nonsmoothness and local minima remain possible. Line searches and stationarity
flags expose failures; AD does not remove this problem.

## Uncertainty and checks

The force conditional and its elimination are exact for the declared model.
The trajectory/material posterior is a **local Gauss-Newton Gaussian**. This
does not repair multimodality or provide calibration. Each fit reports the
marginal Gauss-Newton decrement (screen <0.05), gradient norm, line-search
termination, and Schur/density-identity numerical errors.

Independent Gaussian proposals are checked against the complete nonlinear
marginal posterior. Relative normalized-weight ESS below 0.1 flags a fragile
approximation. This is importance-weight concentration, not MCMC ESS. Passing
does not prove calibration. Invalid proposals retain zero weight and are not
replaced. Predictive invalid draws are counted without replacement; if any
exist, scores explicitly condition on the valid subset. Inference should be
repaired before scientific claims about such intervals.

## Run and handover

```sh
physgauss marginalize --quick --output runs/marginal-smoke
physgauss marginalize --plan configs/marginal_force_study.json \
  --historical-summary examples/force-study/summary.json \
  --output runs/marginal-force-study
# Same committed source, settings, environment and historical reference:
physgauss marginalize --plan configs/marginal_force_study.json \
  --historical-summary examples/force-study/summary.json \
  --output runs/marginal-force-study --resume
```

Checkpointing occurs at completed fit and method boundaries; an interrupted
optimization restarts that fit. Source, settings, observed tensor hashes and
environment are persisted. Initialization, fitted state/precision and paired
draws remain in raw checkpoints. Compact JSON, CSV, figure and self-contained
HTML are suitable for version control; no large checkpoints are needed to
query the portable force marginal. Re-running scientific scene samples needs
the raw joint trajectory/material checkpoints or a fresh execution.

## Mathematical foundations

- [Latent Force Models, Alvarez et al.](https://proceedings.mlr.press/v5/alvarez09a.html)
- [Gaussian process joint input-state estimation, Nayek et al.](https://arxiv.org/abs/1904.00093)
- [Variable projection, O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf)

The nonlinear trajectory/renderer formulation and matched implementation are
our proposed research experiment. The underlying Gaussian integration and
Schur-complement identities are established mathematics, not a novelty claim.
