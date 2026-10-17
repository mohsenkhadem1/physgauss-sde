# Frozen protocol: nonlinear sparse-view posterior comparison

This follow-up changes inference and the point estimator. Geometry, FEM,
staggered six-substep dynamics, rank-three internal kernel, frozen diffusion
weights, priors, six observed images and independent accurate reference data
remain those of `examples/calibrated-noise/comparison`. Each case must reproduce
its saved normalized objective within 1e-7 before sampling. No physical or image
normalizer is omitted, and external force coefficients remain analytically
marginalized at every evaluation.

## Development and final comparison

Development case: `pulse_and_stochastic__4101__neural`. The initial full-gradient
and image-Taylor pilots use 100 warmup and 300 retained transitions per chain;
these are diagnostics, not final results. The Taylor pilot's 300 draws are
insufficient for the complete ESS screen. Seven additional sampler checks pass,
including reversibility, exact high-dimensional Gaussian rotation, nonlinear
known-moment recovery, deliberately inaccurate proposal forces with exact-target
acceptance, physical Taylor derivatives, invalid-state rejection, and bitwise
continuation against an uninterrupted frozen chain. The whole suite has 94 tests.

Final comparison: the original four conditions, three paired seeds 4101–4103,
and floor/constant/neural controls (36 cases). The data are already inspected
benchmark data, not a new blind test. Do not interpret coordinate count as the
number of independent experimental replicates.

Four independent reference initializations and momenta; pooled warmup adapts
one step size for 500 transitions, targeting mean acceptance 0.9. It then freezes.
Each Hamiltonian proposal has a state-independent random duration uniform in
[1,4], rounded to an integer number of steps. The initial step size is 0.3;
dual-averaging bounds are [0.015,0.8]. Seed 99001 plus the fixed case index.
The reference is the regenerated positive Fisher precision at the original MAP;
its accuracy affects efficiency, never the invariant target.

Before final comparison, the development pilot showed remaining slow damping
mixing at durations [1,2] (300 draws: worst Rhat 1.068, minimum bulk/tail ESS
63.2/58.8). The final duration range was extended to [1,4] to traverse this weak
direction. This is sampler tuning on the declared development case, not a
calibration/model change. All other benchmark results are computed only after
this protocol freezes.

Retain 2,000 transitions per chain initially. If **any** monitored quantity fails,
extend the same frozen chains to 4,000, then 8,000, with no additional adaptation.
All retained draws enter estimates. Stop based only on diagnostics, never truth,
reconstruction error, or model ranking. Preserve failures and every diagnostic
stage; no case may be excluded from aggregate comparisons. Finite diagnostics
cannot establish absence of undiscovered modes.

## Exact acceptance with cheap proposal forces

In fixed Gaussian coordinates w, the target is proportional to
`exp(-||w||²/2 + R(w))`. Exact harmonic rotations integrate the Gaussian part;
symmetric momentum kicks use the gradient of an approximate correction. Its
mechanics and state-dependent covariance terms are exact. Only the image NLL
is replaced by its second-order observed Taylor polynomial at the reference
pose, computed independently for each observed image's 12 free coordinates.

The kick/rotate/kick map is reversible and volume preserving. Its full Hamiltonian
Metropolis ratio uses the **original nonlinear renderer**, not the Taylor image
model. Even inaccurate proposal gradients cannot change the target under this
correction. Gradients do not enter the acceptance density. A rejected/invalid
intermediate proposal leaves the old state unchanged. This is a computational
proposal approximation, not a posterior approximation or likelihood change.

## Diagnostics and reporting

Require rank-normalized/folded split Rhat <1.01, bulk and tail ESS >=400, and no
frozen chains for all 1,448 normalized state coordinates, 84 missing-gap position
coordinates, log target, 15 conditional force means, and the profiled quadratic.
For constant/neural controls also monitor 15 conditional force variances and
the two normalizers logdet Q and logdet A: 1,566 total (floor: 1,549, since those
17 additional quantities are analytically constant). Added independent force
draws must not disguise poor trajectory mixing. Record invalid proposals,
energy errors, acceptance, run time and batch-mean MCSE of gap means.

Report each case's posterior mean gap coordinate RMSE, central component 90%
interval coverage/width and full force covariance E[S(z)] + Cov[m(z)]. Compare
with the original MAP and its local intervals under exactly the same model.
No conformal multiplier is applied. These intervals are conditional model
uncertainty; repeated-realization calibration is a separate question. Retain
paired parameter/path/force draws and raw full-coordinate chains in run artifacts.

## Chronological follow-up numerical checks

After inspecting the first completed primary floor case's rejected large-energy
proposals, add independently seeded fixed-smaller-step replicas of the development
floor and neural cases. These are sensitivity checks, not replacements or
retuning of the 36 frozen primary runs. Use step sizes 0.2 (floor) and 0.15
(neural), durations [1,4], seeds 108001 + case index, 500 fixed-kernel burn-in
transitions then 2,000 retained per chain, four chains, no adaptation. Preserve
all diagnostics and compare estimates with the primary runs. Full physical and
image acceptance remains identical. The primary results and their rejected
proposal counts are retained regardless of replica outcomes.
