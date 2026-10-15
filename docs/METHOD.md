# Method implemented in milestone 1

## Physical state and units

The reference object is a 0.65 m cube discretized into eight nodes and six linear tetrahedra. Its four bottom nodes are clamped. Density is known (1 kg/m^3); mass is lumped from reference volumes and never created by rendering. Gravity is -0.6 m/s^2 on the vertical axis. External mass-proportional drag has units 1/s. This is compressible Neo-Hookean FEM, with known Poisson ratio 0.30. There is no inferred contact or plasticity.

For each tetrahedron:

```
F = Ds @ inverse(Dm)
J = det(F)
Psi(F) = mu/2 * (tr(F.T @ F) - 3) - mu*log(J) + lambda/2*log(J)^2
P = dPsi/dF = mu*F + (lambda*log(J) - mu)*inverse(F).T
```

Nodal internal forces are the negative gradient of reference-volume-weighted energy. Time integration is semi-implicit Euler. Nonpositive or nearly collapsed J triggers an explicit failure; no clamp silently turns an inverted element into a valid material. A time-step refinement test checks decreasing trajectory error. This integrator is not exactly energy conserving.

## Rendering

Five barycentric Gaussians per tetrahedron give 30 material primitives. Centers follow the nodal deformation, and each covariance follows `Sigma(t) = F(t) @ Sigma0 @ F(t).T`. Perspective projection uses the pinhole Jacobian; front-to-back alpha compositing produces RGB. Colors and opacity are known for this benchmark. The dense PyTorch implementation is a differentiable reference renderer, not a replacement for a production CUDA splatter.

The support, colors, rest shape, and camera path are known. This is an image-based inverse-dynamics experiment, not yet an unknown-geometry inverse-rendering solution.

## Correlated neural SDE

```
dx = v dt
dv = (internal_force / mass + gravity - drag*v) dt + internal_impulse / mass
```

The stochastic impulse uses standardized independent Brownian coordinates multiplied by sqrt(dt). The three default Brownian modes are shared across material edges through a truncated RBF-kernel basis. The spatial basis is defined in reference material coordinates and normalized to unit edge marginal variance.

A 2 -> 8 -> 1 network with a positive output predicts edge noise amplitudes from relative stretch and radial relative speed, which are invariant to a rigid change of reference frame. Each edge receives a central impulse, and its endpoints receive opposite values. Thus the noise has zero internal net impulse and torque before support reactions. A clamped body can exchange momentum with its support. The noise represents unresolved internal forcing, not thermal noise satisfying a fluctuation-dissipation theorem.

The network is calibrated on 4096 independent synthetic scalar innovation examples and checked on a separate 1024-example set. Test trajectories use an analytic amplitude law and a separate random seed. This explicit simulator supervision is a limitation. The network is not learned from the six monocular frames.

## Inference

The common eight scene coordinates describe log stiffness, log drag, initial top-node displacement (3), and initial top-node velocity (3). Each coordinate has a declared Gaussian prior. Bottom nodes stay at their rest positions.

The physical model estimates those eight coordinates. The SDE model additionally estimates all standardized Brownian increments (60 steps x 3 modes = 180 more coordinates in the default run). Both methods use the same scene priors, input frames, camera path, rendering, and known observation-noise variance. The SDE model has more latent coordinates and a simulator-trained diffusion prior; this is an explicit process-noise ablation, not a capacity-matched comparison against a general neural baseline.

The loss is the Gaussian RGB negative log likelihood plus the standardized normal prior:

```
loss(z) = 0.5 * sum(((render(z, observed_times) - observed_RGB) / sigma)^2)
          + 0.5 * sum(z^2)
```

No hidden RGB frame or hidden node position is passed to the fitting API. Observation noise is added without clipping, to match the likelihood. We use Adam followed by L-BFGS and local Gauss-Newton refinements. Optimization diagnostics include gradient norm and Gauss-Newton decrement; neither alone proves the global MAP was found.

A central finite-difference image Jacobian gives a local precision `H = I + J_image.T @ J_image`, with residuals whitened by observation sigma. Joint scene/forcing samples are drawn from `N(z_MAP, inverse(H))` and physically simulated. This conditions missing intervals on observations on both sides. It is offline smoothing, not causal forecasting.

This Gauss-Newton Laplace approximation drops residual second derivatives, linearizes the local likelihood, and captures one mode. It does not guarantee calibration, resolve unseen geometry, or represent distinct contact-event hypotheses. Posterior samples are not silently discarded if simulation fails. The default 64 samples give noisy interval estimates. MAP objectives are not marginal-likelihood model comparisons.

## Evaluation boundaries

We report componentwise 90% intervals and CRPS on free-node positions, a vector energy score, RGB PSNR, and novel-camera PSNR. Clamped nodes are excluded from uncertainty metrics. Novel-view truth is used only for evaluation. Physical checks include minimum J, net internal force/torque, and support position error. Mechanical energy can change through drag and stochastic forcing.

High image PSNR in this tiny, known-appearance scene is not evidence of real-world reconstruction quality. Coverage within one realization is not an across-scene calibration study. Next tests must vary geometry, material, observation gaps, camera coverage, model mismatch, and stochastic seeds.
