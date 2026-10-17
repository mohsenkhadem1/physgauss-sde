# Neural internal diffusion with marginalized external forces (v0.7)

This extension compares the current fixed full-rank discrepancy with constant and neural balanced internal noise, under identical sparse images and exactly marginalized external-force priors. The numerical integrator and continuous renderer from v0.6 are retained. Neural benefit is a hypothesis to test, not assumed.

The [completed 36-fit results](NEURAL_MARGINAL_RESULTS.md) and [offline report](../examples/neural-marginal/report.html) are published. Structured internal noise improves stochastic point reconstruction, while neural adaptation does not beat the constant control. Most added-noise local posterior diagnostics fail. The separate [true-state calibration audit](../examples/neural-calibration-audit/summary.json) reveals operating-region miscalibration without changing any primary fit or score.

## What is combined

All three models retain the declared velocity-diffusion floor sigma0=0.02 (m/s)/sqrt(s), the same eight scene parameters, known mesh/cameras/appearance, and the same 15-coordinate external force prior. The extra internal diffusion consists of equal/opposite central edge impulses, correlated by a rank-three spatial RBF basis of length 0.45 m. Balance holds before the known supports exchange momentum with the object.

- `floor`: current v0.6 discrepancy and marginalized force.
- `constant`: add the balanced internal diffusion with a constant edge amplitude.
- `neural`: add that same diffusion with amplitude depending on edge strain and preceding-velocity radial speed.

The neural MLP is 2 -> 8 tanh -> 1, with sigma=0.001 softplus(output)+0.0001, 33 weights/biases. It is calibrated on 4,096 independently simulated innovation examples, frozen, and never trained on test motion or withheld images. The constant amplitude is the RMS neural amplitude on a separate fixed calibration feature set, not tuned to reconstruction truth. Constant and neural controls share spatial basis, rank, full-rank floor and force prior. Constant edge amplitude still uses evolving edge directions, so its covariance depends on geometry; only the floor-only model has a state-independent covariance. The network weights, calibration examples/seeds and measured calibration errors are saved.

## Causal state-dependent covariance

Stored u_j is a staggered drift velocity. x_j is the current position, and p_j is the PRECEDING stored velocity (initial physical v0 for the first kick). Evaluate the network at (x_j,p_j), not at the newly updated u_j. This makes the discrete diffusion causal and its density triangular. A state-dependent amplitude in the current unknown velocity would require a different Jacobian and is not used.

Let L_j(x_j,p_j) map the rank-three Brownian increment to free nodal velocity, using central edge directions, amplitudes and inverse mass. For kick duration k_j (first h/2, then h),

    Q_j(z) = k_j [sigma0^2 I + L_j(z) L_j(z)^T].

The positive floor makes Q proper; purely balanced low-rank noise would be singular. The floor remains a declared external/nodal discrepancy component, distinct from the balanced internal part. This is not a reparameterization of the old singular neural SDE.

The centered mechanical residual remains r(z)=G a + eta, a~N(0,I), eta covariance Q(z). G is fixed by the force basis/masses. Conditional on the whole trajectory, Q depends on z but not on a, so exact force integration still applies:

    C(z) = Q(z) + G G^T
    A(z) = I + G^T Q(z)^(-1) G
    a | z,images ~ Normal(A(z)^(-1) G^T Q(z)^(-1) r(z), A(z)^(-1)).

The marginal objective includes r^T C^-1 r /2 + logdet(C)/2, observed image residuals, scene priors, and the same centered-drag/coordinate Jacobians as v0.6. Q, its Cholesky factors, A, and BOTH log determinants are differentiated. Freezing Q or dropping logdet terms would change the model.

## Why marginalization helps

It eliminates the separate force optimization block, preserves a conditional force distribution, and exposes force/trajectory correlations. It does not eliminate nonlinear mechanics or trajectory optimization, create information from missing images, or guarantee faster/better reconstruction than joint fitting.

The distinction becomes more substantial with state-dependent covariance:

    marginal_NLL(z) = min_a joint_NLL(z,a) + 0.5 logdet(A(z)) - (15/2) log(2 pi).

With the old fixed Q and G, A is constant, so joint/profile and marginalized trajectory optima coincide. With Q(z), the volume correction varies with state; simply profiling the force gives a different objective. Marginalization accounts for the range of forces compatible with each trajectory, instead of just its best force.

## Optimization and local uncertainty

A line-searched Gaussian mean-and-covariance information metric is used:

    H_pq = J_image^T J_image + prior
           + (dr/dz_p)^T C^-1 (dr/dz_q)
           + 0.5 tr(C^-1 C_,p C^-1 C_,q)
           - Hessian(log dynamic Jacobian).

The covariance-information term is computed exactly using block-diagonal Q derivatives plus a force-size low-rank correction; no dense third-order derivative of the full C is materialized. Tests compare it with independent full-state AD and dense covariance algebra. This remains a local positive-curvature approximation to the nonlinear posterior, not its exact Hessian or established calibration.

Force uncertainty obeys total covariance:

    Cov(a | images) = E_z[S_a(z)] + Cov_z[m_a(z)].

Unlike v0.6, the conditional covariance S_a(z) now varies and MUST be averaged over trajectory draws. Paired conditional force samples preserve this mixture. Invalid local Gaussian states are counted without replacement. Independent importance-weight diagnostics use the complete nonlinear marginal target, not a frozen-covariance surrogate.

## Fixed comparison

`configs/neural_marginal_study.json` specifies 36 fits: three models, four conditions (deterministic no-load, deterministic pulse, stochastic only, pulse plus stochastic), and three seeds (4101/4102/4103). Every fit sees six 24x24 images at frames 0,3,6,14,17,20 from the same known moving camera. Each seed/model pair shares observed images and the same prior-mean zero-load initial path. There are no extra views supplied to the neural model.

Deterministic truth uses published independent RK4-96 positions. The existing v0.6 noisy seed-4101 images are reused exactly. Stochastic truth uses a separately implemented symplectic Euler-Maruyama reference, evaluated at 192/384/768 substeps per frame with the same coupled Brownian path. Its amplitudes follow the analytic calibration law, not the fitted network. Last-grid coordinate change must be below 0.25 mm; impulse balance, supports and positive deformation determinants are audited. This is a finite-grid consistency check, not a rigorous absolute error bound or experimental validation.

A separate evaluation-only oracle-amplitude/analytic-load audit drives the PRODUCTION staggered integrator with the same Brownian path at six and twelve substeps. Its six-substep coordinate error against the fine reference must be below 0.5 mm. The oracle is never used to initialize or fit reconstruction. Deterministic second-order accuracy alone does not certify stochastic accuracy.

Optimizer budget is 20 iterations, decrement threshold 0.05 and 18 Armijo halvings for all models. Report gap/withheld coordinate error, Gaussian CRPS, local coverage AND width, external-force error/intervals, stationarity, independent 128-proposal weight concentration, and all individual seeds. Three seeds are exploratory; do not claim established uncertainty calibration. Dense/novel camera renders are evaluation only. The network is a simulator-calibrated prior, not evidence that its amplitude can be learned from one sparse real video.

## Reproduction

    python -m pip install -e .
    physgauss check
    physgauss neural-marginalize --quick --output runs/neural-smoke
    physgauss neural-marginalize --plan configs/neural_marginal_study.json --output runs/neural-marginal

Use --resume only with unchanged numerical source, plan, environment and calibration. The quick path exercises all models and covariance calculations but is not a convergence/accuracy result. Historical v0.6 reports remain unchanged.


## Compact handover and publication

The publisher verifies all 36 cases, matching RGB/initial states, source/calibration, saved objectives and point paths, reference gates, force covariance identities, portable force queries and aggregate records:

    python scripts/publish_neural_marginal.py --raw runs/neural-marginal --output examples/neural-marginal --source-ref 08c3281

Use an unused publication directory. Large precision factors stay in raw ignored checkpoints. A new clone can reconstruct one local proposal from compact published solver coordinates, without re-optimizing:

    python scripts/recover_neural_posterior.py --study examples/neural-marginal --case no_load__4101__neural --output runs/recovered/fit.pt

Use the numerical implementation recorded by the publication. Recovery verifies the original image hash, complete normalized objective and final decrement. The covariance-recovery path was checked against the original neural checkpoint. It is still a local approximation; the complete nonlinear-target importance diagnostic remains relevant.

The optional `scripts/prepare_neural_references.py --study runs/neural-marginal` helper may prepare independent stochastic references concurrently with deterministic fits. It calls the identical committed functions and seeds and publishes reference files atomically; it does not change inference or define a new experimental partition.

All 77 checks and all original/new smoke commands passed implementation CI for `08c3281`: https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39037380351 . Neural-weight uncertainty is not propagated; the comparison is conditional on the separately calibrated frozen network.

Reproduce the secondary evaluation-only calibration audit, using the published true frame states and known synthetic amplitude law:

    python scripts/audit_neural_calibration.py --study examples/neural-marginal --output runs/new-calibration-audit

It performs no network training or reconstruction updates. All true-state feature values lie within the original broad training ranges; poor operating-region calibration therefore cannot be dismissed as an out-of-range input check alone.
