# Known-floor correction and conditional posterior sampling

## Why the preceding candidate was stopped

The original seven authorized continuations have all finished. Ten of sixteen
baseline cells pass convergence, including all eight Torus cells and the two
four-frame Bird marginalized cells. The six other Bird cells remain ineligible
for ranking; a successful Actions job does not mean convergence passed.

The new single-Bird run39155024009 (source33ec931) was stopped after an
observed-data-only development audit exposed deep penetration of the known
ground. This was not an inspection of withheld particle truth. In500-draw
development chains, the minimum handle height ranged from-0.32 to-0.70m;
14% of handle/time coordinates were below-0.10m. These chains were not
converged and are diagnostic trajectories, not validated posterior summaries.
The [audit, stopped run and failed sampling trials](../examples/pacnerf-floor-audit)
are retained. The authors' [default configuration](https://github.com/xuan-li/PAC-NeRF/blob/main/configs/default.py)
specifies gravity(0,-9.8,0) and a ground plane at y=0.

The previous smooth, frame-held ground reaction permits penetration. Its
background plate also has no occlusion depth. Adding broad discrepancy noise
can expose those deficiencies. Converging that posterior would not establish
that its uncertainty describes plausible motion.

## New physical target, separately versioned

`floor-conditioned-hierarchical-v2` keeps the original geometry, twelve modes,
load prior, learned-scale priors, frame schedules and exact linear integrals.
It adds two explicit known-boundary assumptions:

* Clip emitted radiance at each camera ray's intersection with the opaque
  y=0 plane, and at camera depth0. A single partial slab per pixel reproduces
  direct per-ray depth clipping without a large slab/splat/pixel tensor.
  The existing projected Gaussian and0.3px anti-alias variance remain an
  approximation; this is not exact ray integration of isotropic world blobs.
* Condition on the softened physical-feasibility event F=1, with
  `P(F=1|path)=exp(-sum softplus(-height/0.002))`. Heights cover all later
  mechanical handles and rendering centers. The2mm scale was declared from
  the known plane and metric geometry, before corrected PAC evaluation.

This is an explicit event likelihood applied to the original joint prior.
The standardized mechanics/noise priors are specified **before** conditioning
on F. We do not claim independently normalized truncated transitions or omit
a parameter-specific truncation constant. F depends on the path, not directly
on the Gaussian force coefficients, so analytic force elimination remains
exact. The joint and marginalized arms still integrate to the same posterior.

This correction changes the posterior. Its samples cannot be pooled with or
resumed from earlier physical targets. It does not replace the graph with MPM,
enforce exact hard contact between frames, or resolve uncertainty outside the
fixed modal space, appearance or camera calibration.

## Inference change

The guide now includes local curvature of the feasibility event. A deterministic
ballistic seed is lifted into the known feasible half-space using only initial
geometry and gravity. Two guide linearizations are followed by optimization
and stationarity checks of the complete nonlinear transported density.

A full observed Hessian supplies a fixed Gaussian reference. Its eigenvalue
floor changes the proposal only. A Schur-complement factorization orders the
five material/noise parameters before the state variables while retaining the
same reference covariance. A conditional state move therefore keeps those
physical parameters fixed.

The sampler alternates Metropolis-corrected state HMC with five scalar Gaussian-
reference elliptical slices for parameters. State HMC has randomized duration
0.6–1.8, target acceptance0.9, and warmup-only step adaptation. Parameter-
dependent operators and the Gaussian guide are cached as exact constants
only during that partial derivative. State-dependent neural covariance,
nonlinear contact, every observed pixel, the floor event and the guide Jacobian
remain in the full target. Tests compare the partial gradient to the full
gradient and reproduce independent correlated-Gaussian moments.

Four independent reference-normal initial states replace0.1-scale starts.
Warmup and retained prefixes preserve the full RNG, adaptation and slice state.
Uninterrupted, warmup-resumed and retained-resumed Gaussian trials are bitwise
identical. Final gates remain Rhat<1.01 and bulk/tail ESS>=400 for every
monitored quantity, with at least2000 retained draws per chain.

## Uncertainty validation and next gate

The [floor-conditioned repeated-data check](../examples/floor-conditioned-interval-validation/validation.json)
uses10000 independent small linear simulations and deterministic posterior
quadrature. Both scale and path are jointly rejection sampled against F;
holding the scale fixed while rejecting paths would validate another model.
Observed coverage is50.39/79.84/89.78/94.62% for50/80/90/95% intervals.
Position PIT and hyperparameter SBC KS statistics are0.00653/0.00636, below
the99% bound0.01628. Doubling the quadrature resolution changes the checked
quantiles by at most0.022mm. The initial2000-dataset result is also preserved;
its low80% coverage motivated extending the same seeded experiment.

The [nonlinear force sampler check](../examples/nonlinear-force-sampler-validation/validation-refscale1.json)
also passes with reference-normal starts. These tests establish arithmetic
and small-model sampling checks, not high-dimensional PAC convergence or
independent real-data calibration.

The short floor-corrected Bird development trial eliminated deep penetration
(no sampled point below-2cm; minimum across the trial about-4.7mm) but did not
converge. It motivated conditional HMC. No corrected Bird reconstruction or
calibration improvement is claimed until the complete single-case check passes.
The larger comparison remains gated on that independently replayed result,
descriptive90% coverage>=80%, CRPS no worse than baseline, and negligible
deep floor penetration. All force/uncertainty comparisons remain exploratory.
