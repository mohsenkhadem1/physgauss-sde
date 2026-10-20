# Sampling the existing soft floor with temporary walls

This changes inference, not the physical contact law or posterior. A reflection
acts on artificial MCMC momentum. It is not a simulated impact of the object.

For fixed material/noise parameters, the parameter-conditioned Gaussian guide
maps state coordinates linearly into physical heights: h = c + A w.
The conditional state density has the form

    p(w) ∝ Normal(w;0,I) exp(R(w)) ∏ sigmoid(h_i/s),  s=0.002m.

R includes the complete dynamics density, nonlinear contact, state-dependent
neural covariance, observed RGB likelihood and guide Jacobian, with the sigmoid
floor factor removed. Introduce

    u_i | w ~ Uniform(0, sigmoid(h_i/s)),  b_i=s logit(u_i).
    p(w | u) ∝ Normal(w;0,I) exp(R(w)) 1{c+A w ≥ b}.

Integrating the auxiliaries restores the original sigmoid product. There is no
replacement by a hard physical floor and no missing parameter normalization.

Residual-gradient kicks alternate with harmonic Gaussian flow reflected at
the temporary linear sampling walls. Intersections are solved analytically;
momentum reflects in the wall normal. This Gaussian-flow component follows
[Pakman and Paninski](https://arxiv.org/abs/1208.4118). A Metropolis correction
uses the augmented Hamiltonian, including numerical energy error. Numerical
invalidity or a reflection cap rejects the proposal. The full non-Gaussian
posterior is not replaced by a quadratic approximation.

The auxiliary walls are discarded before a joint five-parameter elliptical
slice update using the original marginal soft-floor target. Saved log densities
also use that marginal target. Physical contact, force conjugacy and model
priors stay unchanged. The Gaussian reference is learned from discarded
observed-data states; its centre is not claimed to be a stationary mode.

## Validation and current status

The event-driven flow passes energy conservation, reversal through multiple
walls and an independent orthant solution. Logistic augmentation plus residual
Metropolis correction reproduces independent posterior quadrature. The full
alternating state/parameter sampler reproduces correlated joint posterior
moments computed by a separate Gaussian-conditioning integral. Warmup and
retained-prefix resumption are bit-identical. Neural joint/marginal fixture
tests verify the physical height map and residual gradient.

All182 regression tests pass. The actual Bird preflight height error is6.4e-16m
and residual-density identity error1.6e-12. Its empirical-covariance development
trial was stopped in warmup: small conditional steps remain necessary. A guide
without floor curvature is the next proposal experiment. No convergence is
claimed. Production requires at least2000 retained draws/chain, Rhat<1.01,
bulk/tailESS>=400 for every monitored state, parameter, conditional force and
normalizer, followed by independent uncertainty evaluation. The larger
comparison stays gated.
