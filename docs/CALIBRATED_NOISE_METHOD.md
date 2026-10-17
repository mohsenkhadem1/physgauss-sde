# Calibrated neural diffusion and uncertainty (v0.8.1)

Read `CALIBRATED_NOISE_PROTOCOL.md` for the fixed experiments, corrections and
fresh test split. `CALIBRATED_NOISE_RESULTS.md` reports the final measurements.
The underlying model and exact force elimination are derived in
`NEURAL_MARGINALIZATION.md`.

## What the network learns

The 33-parameter network predicts an edge's force-noise amplitude. Its two
inputs are signed strain and radial relative speed, transformed to smooth
magnitudes and divided by fixed scales [0.10, 0.50]. The architecture is
`Linear(2,8) -> tanh -> Linear(8,1) -> .001 softplus + .0001`.

Training uses 12 independent simulated motions and an equal number of broad
random feature states: 9,576 feature pairs total. Each has 128 independently
simulated scalar Gaussian innovations. Minimizing

\[
\frac{1}{N}\sum_i\left[\frac{\widehat{E[\epsilon_i^2]}}{2\sigma_\psi(s_i)^2}
                       +\log\sigma_\psi(s_i)\right]
\]

uses innovation observations rather than analytic amplitude regression labels.
Four other trajectories validate it. Analytic amplitudes remain known in this
synthetic simulator and are used for evaluation. The constant comparator is
the innovation standard-deviation MLE on training operating-region states.
Weights freeze before reconstruction. Network-weight uncertainty is absent.

Edge impulses are central and equal/opposite before support clamping. The
spatial covariance uses the exact rank-three projection already published in
v0.7, now frozen as source constants because its eigenvalue cut crosses a
triple eigenspace. This is a fixed covariance approximation, not the complete
isotropic RBF kernel. A declared independent full-rank velocity floor remains.

## Marginalized mechanics and exact observed curvature

For a proposed trajectory/material state z, force coefficients a are Gaussian:

\[
r(z)=Ga+\eta,\quad a\sim N(0,I),\quad\eta\sim N(0,Q_\psi(z)),
\qquad C(z)=Q_\psi(z)+GG^T.
\]

The normalized marginalized dynamics loss is
`0.5*r^T*C^-1*r + 0.5*logdet(C) + constant`. Image likelihood, parameter
priors and the staggered transition density Jacobian complete the target.
Both models retain force marginalization; this study changes noise calibration.

Let A=I+G^T Q^-1 G and a*(z) be the exact Gaussian conditional mean. Then

\[
\ell_m(z)=\ell_j(z,a^*)+\tfrac12\log|A(z)|-\tfrac K2\log(2\pi),
\]
\[
H_m=H^j_{zz}-H^j_{za}A^{-1}H^j_{az}
                         +\tfrac12\nabla_z^2\log|A(z)|.
\]

The audit assembles local mechanics Hessians and exact observed-image
curvature, then applies this Schur identity. The volume term includes both
`tr(A^-1 A_,pq)` and `-tr(A^-1 A_,p A^-1 A_,q)`; omitting the latter is wrong.
Dense AD on a small model, full-target Hessian-vector products and refined
central gradient differences check the result. Nonpositive observed curvature
is rejected rather than clipped. Its Gaussian is still only a proposal.

Numerically empty depth slabs use the analytic emission factor
`phi(tau)=(1-exp(-tau))/tau`, with `phi(0)=1` and a Taylor evaluation near zero.
The old source/guarded-density ratio had incorrect emission derivatives there.
Tests check value, first derivative and second derivative against a closed-form
single-color volume solution, including exactly zero density.

## Two different uncertainty checks

Four pCN chains with Gaussian-reference independent refreshes apply Metropolis
correction to the full nonlinear marginalized target. The Gaussian only
preconditions moves. Warmup adaptation ends before retained sampling. All
state coordinates, gap positions, target density and conditional force means
are monitored using rank/folded split Rhat and bulk/tail ESS. A diagnostic pass
does not prove global exploration or real-world Bayesian calibration.

Force moments use `E[S_a(z)] + Cov[m_a(z)]`, with S_a(z)=A(z)^-1. Conditional
force draws stay paired with their trajectory. Independent force draws are not
used to conceal poor trajectory exploration in convergence diagnostics.

Separately, each of 19 calibration motions supplies one maximum standardized
error across 84 missing-gap position coordinates. Its 18th order statistic
sets a frozen 90% split-conformal prediction multiplier. Twenty fresh motions
test whole-gap coverage, with Wilson uncertainty based on 20 independent
trials. This is stricter than coordinate-wise coverage. It is a repeated-case
prediction-set property under exchangeability, not a corrected posterior or
force/material uncertainty guarantee. No test case tunes the network or band.

## Reproduction and runtime

```sh
python -m pip install -e '.[performance]'
physgauss check
physgauss calibrated-noise --plan configs/calibrated_noise_frozen_basis_study.json --paired-source examples/neural-marginal --output runs/new-calibrated
python scripts/audit_calibrated_posterior.py --study runs/new-calibrated/comparison --case pulse_and_stochastic__4101__neural --output runs/new-posterior-audit
```

The optional opt-einsum dependency changes contraction order, not equations.
On the recorded probe, four-target evaluation was about 2.2 times faster;
maximum target/gradient differences were 9.1e-13 / 2.7e-13. Audit settings record
its version and greedy strategy. Primary CPU workers use the default path;
all published targets and uncertainty factors are independently reconstructed
locally with explicit cross-runtime numerical tolerances.

The primary comparison and validation can also run through their separate
GitHub workflows. Both use frozen network weights and a serialized prior
initializer, with exact original RGB for the 36 matched fits. Calibration jobs
form a barrier before fresh held-out validation jobs. Worker artifacts preserve
inputs, reference states, solver coordinates and diagnostics; full factors are
regenerated without optimization using `materialize_validation_precisions.py`.
