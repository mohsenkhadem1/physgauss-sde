# Material and boundary Gaussians for deformation inference

## Scientific question

Given a static Gaussian reference and sparse partial observations, infer the
locations, types, parameters and histories of boundary conditions. Use them to
predict hidden deformation with calibrated uncertainty. Material uncertainty is
a subsequent extension; it must not be confused with unknown supports or loads.

There are two functional Gaussian families in reference/material coordinates.
Gaussian kernel extent describes a spatial field, NOT uncertainty about the
estimated parameter. Posterior uncertainty is a distribution over kernel and
physical parameters.

## 1. Object/material kernels

For a reference body Omega, let g_i(X) be positive Gaussian kernels and use a
partition of unity w_i(X)=g_i(X)/sum_j g_j(X). Define a constitutive energy

    Psi(F,X;theta) = sum_i w_i(X) Psi_i(F;theta_i)
    P = d Psi / d F
    rho_0 d²u/dt² = Div P + rho_0 g,   F = I + grad_X u.

The component energies need an appropriate constitutive model. Positive moduli
alone do not guarantee every desired stability property. Frame indifference,
compressibility, dissipation and mesh convergence must be tested. The initial
configuration may contain stress; the initial beam benchmark explicitly assumes
a stress-free reference and omits gravity.

Material parameters are attached to the reference object. Appearance Gaussians
can follow the same deformation using a first-order local covariance pushforward
Sigma_i(t) ~= F_i Sigma_i(0) F_i^T. Appearance and mechanical discretization can
have different resolutions; opacity does not specify mass density or material
connectivity. The beam implementation currently has three coarse material
kernels with fixed identical EI=0.8 N m² and denser illustration splats.

## 2. Boundary kernels

Boundary kernels live on the actual reference surface Gamma, including hidden
parts. For a patch j, restrict a Gaussian to Gamma and normalize by surface
measure:

    b_j(X) = exp(-d_j(X)^2/2) / integral_Gamma exp(-d_j(S)^2/2) dA(S).

Patch coordinates should follow surface topology rather than accidentally join
nearby but disconnected surfaces. Geodesic or surface-chart kernels are possible.
The boundary reference patch advects with the material. An environmental anchor
or tool frame can move independently; these are different state variables.

A useful restricted family of traction laws is

    traction(X,t) = sum_j b_j(X) [a_j(t)
                         - K_j (u(X,t)-d_j(t))
                         - C_j (du/dt-dd_j/dt)].

Here a_j is a total force vector, d_j is anchor displacement, and K_j,C_j are
positive semidefinite integrated attachment stiffness/damping matrices. With a
surface-normalized b_j, traction has force-per-area units. Reference and current
surface tractions require the correct area/normal transformation; the 3D formula
must not silently mix Cauchy and first-Piola tractions.

This law represents free, loaded, and compliant regions. A hard prescribed
displacement is imposed as a constraint rather than approximated by arbitrarily
large stiffness. Its reaction is an output. Contact requires nonpenetration and
complementarity (and a friction law if needed), not an unconstrained Gaussian
force. Type uncertainty is categorical; uncertain numerical parameters should
not be used to interpolate indiscriminately between incompatible physical laws.

Patch center, extent, type, anchor and force history are separate unknowns.
Visual features propose them; observed motion and the mechanical model refine
their posterior. Patches should be few, spatially coherent and persistent unless
there is evidence of attachment/release. A PDE plus completely unrestricted
boundary loads does not make the inverse problem identifiable.

## 3. Time-dependent inference and flow matching (proposed next stage)

Let eta contain transformed material and boundary parameters and load-history
coefficients. E.g., positive scalar stiffness uses exp(eta_k); full positive
definite matrices can use Cholesky factors. Let c encode only permitted images,
camera poses, masks, timestamps and known actions. The desired target is

    p(eta, u_0:T | observed images, static reference, cameras, known actions).

Conditional flow matching could amortize this posterior. With training pairs
(eta_1,c) generated across many simulated systems, sample eta_0~N(0,I) and
tau~Uniform(0,1), construct eta_tau=(1-tau)eta_0+tau eta_1 and train

    L_CFM = E ||v_phi(eta_tau,tau,c) - (eta_1-eta_0)||².

At inference, integrate d eta_tau/d tau=v_phi from tau=0 to1. Tau is generative
time, not the physical time t. The physical PDE evolves u(t). Categorical patch
types require a separate categorical model or discrete generative method.

The encoder learns a conditional distribution from many examples. Flow matching
alone does not identify material properties from one sparse clip. Simulator
residuals, observation consistency and held-out coverage must be checked. An
amortized posterior conditioned on the images must not be treated as a prior and
multiplied by the same image likelihood again. Exact-target correction requires
a valid proposal/importance or MCMC construction. Neither DINO nor a
flow-matching network has been trained in this first benchmark.

Sequential updates can filter the current state and smooth a short recent
window, retaining slow material/fixture beliefs and faster loads/contact states.
Known physical manipulations add boundary information. Virtual edits produce
counterfactual predictions but do not supply new evidence about real materials.

## 4. How force marginalization fits

Conditional on a boundary configuration beta, an affine Gaussian load model
gives r=B_beta a+epsilon, a~N(0,K), epsilon~N(0,Q). Therefore

    r | beta ~ N(0, Q+B_beta K B_beta^T).

Integrate a, retain uncertainty over beta, and reconstruct the conditional load
posterior when needed. Unknown contact switches and boundary types generally
remain outside the Gaussian integral. The beam uses an equivalent observation
form y=H_beta a+noise and exactly enumerates 24 beta values. For general nonlinear
mechanics, linearization must be declared; this exact linear result does not
automatically extend to nonlinear image likelihoods or nonlinear contact.

## 5. Implemented beam benchmark

The reduced model is a one-metre Euler-Bernoulli beam under slowly varying
quasistatic loads, bending in one plane. It has natural free ends and a compliant
Gaussian support, not a hard clamp. In weak form, for test function v,

    integral EI(X) u'' v'' dX + integral k_s b_s(X) u v dX
        = integral F(t) b_f(X) v dX.

In this reduced model b_s,b_f have units 1/m, k_s is N/m, and F is N. Both kernels
are Gaussian densities normalized over [0,1]. They have tails; support/load
centres are outside the observed interval, but the kernels are not compactly
supported. Fixtures and forces are not directly shown to the estimator.

The unknown dictionary has 3 support centers x 2 widths x 2 support stiffnesses
x 2 load centers =24 hypotheses, uniformly weighted before observations.
Eight force amplitudes at t=1,...,8 s have a known zero-mean GP prior conditioned
on F(0)=0: std0.06 N, correlation length2 s. Signed forces allow either transverse
direction. There is no nonnegative-force or contact interpretation.

Training observations are noisy 2D displacements of known reference Gaussian
centres under a smooth, known orthographic camera path (400 px/m, noise0.25 px).
The one-plane motion assumption leaves only one informative image direction.
This does not yet address full 3D monocular depth ambiguity or correspondence
estimation. RGB figures use an artificial reference-coordinate visibility mask;
they are illustrations and are not used for pixel-likelihood fitting.

Sparse inputs: 3 times (1,4,8 s), 3 centers per image. Richer inputs:5 times
(1,2,4,6,8 s), 9 centers per image. Common tracks share the same noise realization.
The reference uses80 cubic-Hermite elements; inference uses40. Eight-point
quadrature integrates material and boundary kernels. The physics audit compares
40/80/160 elements and checks force, moment, energy and unit-load normalization.
Independent tests compare to an analytic uniformly loaded cantilever and dense
Gaussian conditioning.

Comparisons:

- Inferred boundaries: full mixture over all24 candidates, forces marginalized.
- Known boundaries: true boundary configuration supplied; force amplitudes still
  inferred from the identical noisy observations. This is the informative oracle.
- MAP boundary plug-in: select one candidate, retaining only its force variance.
- Wrong fixed boundary: always use dictionary entry zero.
- Fully known boundaries AND force: reference solution by construction; no
  meaningful statistical ranking, so it is not included in the error table.

Uncertainty validation uses1000 independent prior-predictive cases. The primary
query is the hidden free tip at missing time5 s, plus the force at that time.
Equal-tail mixture intervals are computed by numerical inversion of the exact
mixture CDF. All50/80/90/95% coverages and Wilson intervals are reported. The gate
requires90% coverage within3 binomial standard errors and PIT KS<=1.63/sqrt(N)
for both query types and both learned/known-boundary models in both designs.
These are marginal checks, not a family-wise test, simultaneous path coverage,
or guarantees for each possible fixture. Discrete credible-set coverage is
reported with the actual posterior mass (which can exceed90%).

Separate400-case stress arms test off-grid boundaries, shorter true load
correlation (0.8 s), and both together without changing inference. The first
combined arm was declared before results. Isolated arms were added after its
undercoverage to diagnose the source. Stress outcomes are exploratory rather
than a second untouched calibration-validation set.

## Reproduction

From the repository root, with Python, NumPy and Matplotlib installed:

```sh
python -m unittest discover -s tests -p test_beam_boundary.py -v
python experiments/run_beam_boundary.py --config configs/beam_boundary.json --output examples/beam-boundary
```

Code, protocol, compact metrics, figures and compressed JSON datasets are
committed. The datasets contain evaluation ground truth alongside observations;
the estimator receives only the observations and the declared prior. Truth is
used exclusively for generation, evaluation and the explicitly labelled oracle.
Local execution is CPU only. `.github/workflows/beam-boundary.yml` reproduces the
tests and study on a standard free CPU runner with read-only repository access.

## Related primary sources

- [Inverse boundary identification with material uncertainty](https://doi.org/10.1016/j.camwa.2016.12.011)
- [Multiple-experiment material and boundary identification](https://doi.org/10.1016/j.ijmecsci.2023.108600)
- [Differentiable 3DGS-SPH boundary/material identification, October2026 preprint](https://arxiv.org/abs/2610.04336)

These precedents mean that Gaussian boundary parameterization alone is not a
novelty claim. The proposed research contribution is sparse-view, sequential
boundary inference with validated uncertainty and informative interventions.
