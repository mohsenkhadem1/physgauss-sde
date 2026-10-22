# Continuous numerical reference for a small compliant-fixture inverse

The two-parameter **conditional slice** of the manufactured compliant-fixture problem now has an independently integrated posterior and continuous MCMC validation. The original four-parameter problem remains unvalidated: its Gauss–Legendre625 and Clenshaw–Curtis83521 grid failures are preserved. No four-dimensional posterior run or neural-network expansion is presented here.

This uses the exact same finite linear dynamics, saved twelve displacement observations, 20 µm noise, fixed applied forces, and original intrinsic-uniform prior. Young's modulus E and finite Gaussian fixture stiffness K remain unknown; fixture center and width are fixed to the published manufactured truth for this explicitly conditional diagnostic. This is not the four-dimensional marginal posterior. K is finite compliance, with no exact pin; all fifteen nodal degrees of freedom remain free.

## Independent integration and continuous MCMC

Adaptive Gauss–Kronrod21 cubature is independent of the samplers and uses two declared relative tolerances, 10⁻⁵ and 10⁻⁷. They integrate normalized intrinsic prior mass, compute marginal CDFs at 33 declared knots, root-solve the 5/50/95% quantiles, and integrate full future-A/reset-B first and second moments. Their maximum CDF difference is 1.81×10⁻¹²; quantiles agree at the declared 18-step bisection resolution; field-moment differences are at most 5.4×10⁻¹⁷ m.

| Parameter | Adaptive 5/50/95% quantiles | Continuous MCMC 5/50/95% |
| --- | --- | --- |
| E, Pa | 3365.05 / 3891.72 / 4668.30 | 3367.81 / 3882.63 / 4650.78 |
| Fixture K, N/m | 5.58998 / 12.77382 / 26.72264 | 5.61491 / 12.88227 / 26.79592 |

Eight independent MCMC chains start from the prior, discard 500 warmup steps, and retain 2000 unthinned draws per chain. A fixed mixture of a pilot-fitted independence proposal, symmetric local reflected moves, and whole-prior moves uses exact Metropolis–Hastings correction. The proposal changes sampling efficiency, not the posterior target.

All monitored intrinsic coordinates and log likelihood pass the frozen rank/folded split R-hat and bulk/tail ESS criteria: maximum R-hat 1.00220; minimum bulk ESS 2763; minimum tail ESS 3095; no frozen chains. Against adaptive integration, maximum CDF error is 0.01684 and unit-coordinate quantile error 0.00473. Future-A mean/std field errors are 2.554/2.274 µm against a 4.716 µm tolerance; reset-B errors are 2.032/1.806 µm against 3.753 µm. The analytic truncated-Gaussian sampler check also passes, with CDF error 0.01548 and quantile error 0.00498.

These are finite numerical checks, not a rigorous proof of continuous integration accuracy, exclusion of every possible mode, or calibrated physical intervals. One manufactured truth falling inside an interval cannot establish coverage.

## Particle resolution and preserved failures

The first fixed six SMC runs, three seeds at each of 512/2048 particles, did **not** meet the combined gate. Two 512-particle comparisons against adaptive integration failed; ten of fifteen seed/count pair comparisons failed. All three 2048 runs individually matched the reference, but some cross-run forecast differences still exceeded the declared tolerance. Uniform resampled weights and a high weight ESS alone were not accepted as convergence.

A separately frozen numerical-resolution diagnosis increases only particle count, to 8192/32768 with the same three seeds, algorithms, physical inputs, and CDF/quantile/forecast criteria. Its 180-second cap preserved four completed runs and the partial fifth attempt. A new parity-tested observation-only evaluator then stops at acquired frame24, stores only frames12/24, and uses fixed1024 chunks. This changes neither the float64 finite model nor the observation likelihood. A separate continuation reuses all four completed results and completes only the same two remaining seeds; no adaptive integration or MCMC is repeated.

Final higher-particle result: **all six fixed runs pass** against both adaptive integration and MCMC, and **all fifteen seed/count comparisons pass** the unchanged criteria. Maximum high-SMC CDF error against adaptive integration is 0.01426; maximum unit-quantile error is 0.004715. Worst future-A/reset-B mean-field differences are 1.334/1.061 µm and standard-deviation differences 0.845/0.678 µm. Maximum pairwise CDF difference is 0.01733 and unit-quantile difference 0.00836. The two-slot completion takes 44.51 seconds and terminates normally; all six slots are present, including the four reused completed outputs. This satisfies the declared numerical validation for this conditional two-dimensional target, rather than repairing the four-dimensional grid or establishing physical calibration.

The reporting failure in the first driver is also preserved: a NumPy boolean could not be serialized, after both adaptive integrations and the first SMC completed. The corrected continuation restores those hash-bound outputs. Dedicated NumPy-scalar and failure-save tests cover that correction. No seed was redrawn or outcome discarded.

## Artifacts and limits

Frozen protocols are under `runs/operator-dev/compliant-continuous-reference-v01`, `compliant-continuous-reference-v02`, `compliant-smc-resolution-v03`, and `compliant-smc-resolution-v04`, with corresponding compact results under `runs/gaussian-campaign-v01`. The final compact receipt is `runs/operator-dev/compliant-smc-resolution-v04/result-summary.json`; the complete six-run comparison is `runs/gaussian-campaign-v01/compliant-smc-resolution-v04/summary.json`. Eight source/algorithm tests cover exact forward/likelihood parity, the conditional prior, independent analytic integration, MH density correction, scalar/failure serialization, observation-only/chunk parity, and causality. Raw particle arrays, chain draws, observations, and moment fields remain local.

This verifies inference under one stipulated finite linear manufactured model. It does not validate nonlinear dynamics, Isaac behavior, general unknown boundary types, sparse RGB inference, population calibration, or active probing. Gaussian shape covariance still describes transported physical shape; posterior weights and chain variation represent uncertainty separately. The original four-dimensional material/fixture ambiguity is not resolved by conditioning two unknowns to their manufactured truth.
