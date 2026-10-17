# Accurate dynamics and continuous visibility (v0.6)

This experiment addresses two numerical faults found in the simulation audit: first-order integration bias and discontinuities when globally sorted splats exchange depth. Historical implementations and results remain available without replacing their measurements.

## Dynamics and the probability density

Use uniform step h, positions x_j, and staggered drift velocities u_j. Let k_0=h/2 and k_j=h thereafter. With previous velocity p_0=v_0 and p_j=u_{j-1}, the update is

    (1 + gamma k_j/2) u_j = (1 - gamma k_j/2) p_j
                               + k_j [a_theta(x_j) + B_j a] + sigma sqrt(k_j) epsilon_j
    x_{j+1} = x_j + h u_j.

Here a_theta includes Neo-Hookean elasticity and known gravity, B_j maps the 15 whitened GP load coefficients to nodal acceleration, and epsilon is independent standard normal over free coordinates. Supports remain clamped. The first half kick initializes the staggered velocity. On deterministic smooth motion this is second order; it does not certify higher-order stochastic accuracy. Reconstructed integer-time velocities are centered finite-difference estimates, not exact instantaneous SDE velocities.

Stack r_j=(1+gamma k_j/2)u_j-(1-gamma k_j/2)p_j-k_j a_theta(x_j), G_j=k_j B_j and Q_j=sigma^2 k_j I. Gaussian elimination remains exact:

    C = Q + G G^T
    a | trajectory ~ Normal(A^{-1} G^T Q^{-1} r, A^{-1})
    A = I + G^T Q^{-1} G.

The residual-to-velocity map is triangular. Its log determinant is D sum_j log(1+gamma k_j/2), D=12. The trajectory negative log density SUBTRACTS this term as well as the existing coordinate-scale determinant. Omitting it biases inferred drag. Its exact negative log-determinant curvature is included in the otherwise Gauss-Newton approximation. Tests compare the direct determinant, innovations, derivatives, and explicit/profile/marginal density identities. No hidden jitter or covariance floor is added to force elimination.

The discrepancy remains the declared full-rank independent nodal impulse model with sigma=0.02 (m/s)/sqrt(s). It is distinct from the historical balanced neural internal SDE. Changing the discrete integrator also changes its finite-step prior; equal sigma does not make the discrete distributions identical.

## Continuous depth integration

The legacy EWA screen footprint alpha_g(p) is retained, with optical depth tau_g(p)=-log(1-alpha_g(p)). Each primitive has a normal depth profile from its camera-space covariance. Integrating that profile over fixed depth slabs gives fractions w_sg, summing to one, including two outer infinite slabs. Slab extinction is sum_g w_sg tau_g, and the slab colour is its extinction-weighted mixture. Beer-Lambert absorption/emission is composed front to back by slab. Whole primitives are never sorted.

This is permutation invariant and continuous as primitive depths cross. Tests check these properties, finite-difference gradients at ties, bounded radiance, and analytic constant-colour transmission. The same alpha footprints imply unchanged total opacity, but overlapping colours differ from sorted alpha blending. It is a projected-footprint volume surrogate: depth and footprint correlations are approximated, and it is not exact 3D Gaussian ray integration or a mass-conserving optical material.

Production inference uses 128 slabs and reference images use 512; a 1024-slab convergence check and independent RK4-96 motion provide numerical error checks. No moving depth bounds, visibility sorting or depth-tie perturbations are used. The new images therefore define a changed image-formation likelihood. Renderer controls use the SAME new data; the hard renderer deliberately approximates it.

## Fixed comparison protocol

`configs/accurate_reference_study.json` specifies two deterministic conditions (no load and analytic external pulse), one geometry and realization. RK4-96 reference positions are loaded from the published simulation audit; this is an independent integration routine using the same FEM force law, not independent validation of that constitutive law. RK4-48/96 agreement is about 2.1e-10 m. No converged stochastic reference is claimed.

The original six observed-image noise residuals are recovered from the original published RGB and hard renders of RK4 poses, then added to the new volume-rendered images. Extra views use fixed seeds. All methods share the same images, prior constants and frame positions at initialization. Initialization is a zero-load corrected rollout at the scene prior mean: no reference positions, true forces, withheld images, or truth-selected warm starts enter fitting.

Six-view comparisons include alternating, joint, and marginalized solvers with corrected dynamics/visibility; marginalized component controls are legacy, integrator-only, and renderer-only. Each receives the same maximum 20 iterations, Armijo backtracking and decrement threshold 0.05. Five view schedules repeat the marginalized corrected fit: six and twelve moving-camera timestamps, six timestamps from two known cameras, and all 21 timestamps from one or two cameras. The original missing gap becomes observed in dense schedules; its score remains a common-window comparison, not withheld performance in those rows. The +200-degree evaluation camera is never observed.

Position intervals are the exact linear pushforward of the LOCAL Gauss-Newton trajectory Gaussian. Force uncertainty combines conditional covariance and variation of conditional means over 128 trajectory draws. Independent 128-proposal nonlinear importance diagnostics are reported for the six primary corrected solver fits. These are diagnostics, not corrected posterior samples or a calibration study. Invalid Gaussian states are reported without silent replacement.

Raw checkpoints are under the chosen ignored runs directory. Compact inputs, point trajectories, force distributions, source/environment hashes and offline report are published separately. A source fingerprint prevents resuming with changed inference code or reference inputs.

## Reproduction

    python -m pip install -e .
    physgauss check
    physgauss accurate --quick --output runs/accurate-smoke
    physgauss accurate --plan configs/accurate_reference_study.json --output runs/accurate-reference
    # After interruption, add --resume with unchanged code, plan and environment.

The quick command only exercises the pipeline; it is not an accuracy or convergence result.


## Publishing a verified handover

After the full comparison completes, run:

    python scripts/publish_accurate_study.py --raw runs/accurate-reference --output examples/accurate-reference --source-ref c35b5a1

Use an unused publication directory. The publisher requires all 20 declared fits and verifies committed numerical source, regenerated inputs/noise, shared initialization, saved objectives, solver-state trajectories, Gaussian elimination identities, and force covariance. It writes an offline report with actual images, accuracy/convergence/view plots, and a renderer continuity probe at the published historical stalled pulse pose. That last probe is post-fit diagnosis, not a setting-selection experiment; exact hard-sort jumps depend on rounding and platform. Publication hashes normalize text line endings.

CI for implementation `c35b5a1` passed all 68 checks and all end-to-end smoke commands, including the accurate-reference pipeline: https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39033558713 .
