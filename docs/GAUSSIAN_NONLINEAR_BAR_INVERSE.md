# Small nonlinear static bar inverse

This is a new controlled inverse for **Young's modulus E and finite spring stiffness K**, using a homogeneous compressible Neo-Hookean bar with known support location and law. It is separate from the earlier conditional linear example, the unresolved four-parameter Gaussian-fixture posterior, and the localized Gaussian-boundary mesh failures. It contains no neural network.

The 0.24 × 0.12 × 0.08 m prism has a fully consistent axial spring on its left face, uniform known nominal traction on its right face, and exact transverse symmetry conditions. Finite-strain homogeneous deformation is checked independently against static sparse finite elements on 48/384-tetrahedron meshes. Its analytical traction law is evaluated by independent nested Brent roots and a parity-tested vectorized Lambert-W/Newton implementation; neither uses a linearized deformation surrogate.

The synthetic design is frozen before generating any observations. Independent log-uniform priors cover E=4500–9000 Pa and K=100–400 N/m. The fixed truth is E=6000 Pa, K=220 N/m. Four acquired channels measure left/right centerline displacement under known 1 N and 3 N forces, with independent 200 µm Gaussian noise and seed 2026121300. Left spring translation identifies K; the additional material extension identifies E. The likelihood uses the four raw channels once, preserving the induced correlation if differences are considered for explanation.

The held 2 N response is **force interpolation, not future forecasting**. All six SMC particle arrays and eight MCMC chains are frozen before its truth is scored. Predictive intervals below describe latent displacement; measurement noise is reported separately. No population coverage or physical-calibration claim follows from one manufactured example.

## Numerical checks fixed before observation

- 36 sparse-FEM/independent-oracle checks at the truth and eight prior points under 1/3 N, before observation generation; 56 further checks at posterior quantile combinations and the held truth after posterior freeze.
- Two adaptive Gauss–Kronrod posterior integrations, relative tolerances 10⁻⁶/10⁻⁸, complete marginal CDF/quantile and held-response moment comparisons.
- Three independent SMC seeds at 8192 and 32768 particles, including every seed/count comparison, original-temperature/ESS checks, and exact prior-preserving Metropolis rejuvenation. Uniform final weights alone do not establish convergence.
- Eight independent prior-started MCMC chains, 1000 warmup steps and 4000 unthinned draws each; rank/folded split R-hat <1.01 and bulk/tail ESS ≥400 for both intrinsic parameters, log likelihood, physical parameters, and held latent displacements.
- A single 300-second global CPU budget with two threads. Any interruption or failed check remains in the output; there is no retry, seed selection, physical retuning, or hidden removal of a case.

## Results

The corrected numerical run passes all predeclared checks in **38.85 seconds**, with a 300-second cap. All six SMC runs match the independent reference and MCMC, and all fifteen seed/count comparisons pass. Maximum SMC CDF error against the reference is 0.00779; maximum pairwise error is 0.01233. Eight prior-started chains pass the intrinsic, physical-parameter, likelihood, and held-response checks: maximum rank/folded split R-hat 1.000815; minimum bulk/tail ESS 10453. All 92 independent sparse-FEM checks pass, with maximum full-field position discrepancy 6.26×10⁻¹³ m.

**Numerical agreement does not establish calibration.** For this one fixed noise realization, both true parameters fall outside their nominal 90% posterior intervals, and one of two held latent displacements does too. The data and seed remain unchanged.

| Parameter | Fixed truth | Adaptive posterior 5 / 50 / 95% |
| --- | ---: | --- |
| E, Pa | 6000 | 5600.54 / 5775.52 / 5962.28 |
| K, N/m | 220 | 220.714 / 225.900 / 231.336 |

| Held 2 N channel | True displacement | Posterior mean | Absolute error | Latent 90% interval width | Truth in interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Left | 9.0909 mm | 8.8515 mm | 0.2394 mm | 0.4197 mm | No |
| Right | 17.6661 mm | 17.7727 mm | 0.1066 mm | 0.4025 mm | Yes |

The left interval is [8.6412, 9.0609] mm; the right is [17.5714, 17.9740] mm. These are latent-response intervals from the fixed MCMC draws. Adding the known 200 µm sensor noise gives observation standard deviations 0.2364/0.2343 mm, without a population coverage claim.

The physical response is nonlinear at the measured scale. At the true parameters, right-end displacement differs from the small-strain tangent prediction by 0.0600, 0.2418, and 0.5483 mm at 1, 2, and 3 N. The largest exceeds the observation noise by a factor of 2.74.

## Preserved numerical failure and repair

Version 1 stopped before any sampling: CDFs, quantiles, and means agreed, but right-response standard deviations differed by 187.3 nm, exceeding the unchanged 100 nm numerical gate. Raw second moments were integrated at a scale set by the much larger squared mean; subtraction magnified their relative quadrature errors roughly 20,700-fold. Separately integrated normalization also contributed. This was a numerical-moment failure, not evidence that the sampler failed.

Version 2 reuses exactly the same hash-bound four observations, pre-FEM checks, CDFs, quantiles, priors, noise, and forces. Both tolerance passes use the same frozen tight-reference mean as a center and the unchanged 200 µm noise as numerical units. A single vector integral contains normalization, centered first moments, and centered second moments. Componentwise quadrature errors propagate through normalization, variance, and standard-deviation brackets. These estimates are heuristic numerical diagnostics, not rigorous guarantees. The original 100 nm gate is retained; estimated-error and mean-parity gates are additionally checked. Repaired two-tolerance standard deviations differ by less than 4.4×10⁻¹⁹ m. No seed, datum, scientific parameter, or inference criterion was selected again.

## Scope and reproduction

The analytical solution is exactly representable by the tetrahedral elements, so successful FEM verification is a homogeneous patch check, not spatial qualification of arbitrary Gaussian fixtures, concentrated loads, contacts, or dynamic deformation. Support position and constitutive law are known; uncertainty is only in two scalar stiffnesses. Rendering, sparse-view inference, general unknown boundary conditions, active probing, Isaac behavior, and population calibration remain outside this proof.

Sources: `experiments/campaign_nonlinear_bar_inverse.py` and `experiments/run_nonlinear_bar_inverse.py`. Original frozen design: `runs/operator-dev/nonlinear-bar-inverse-v01/protocol.json`, SHA256 `5be38c0085c7e7415f9e3bc93b4392cae943b1c0c15554938de59be84de4f427`. The separately frozen centered-repair protocol is `runs/operator-dev/nonlinear-bar-inverse-v02/protocol.json`, SHA256 `04e9baeb0116f2007536f83c3befb2dd920a71d7a48b2fc11fc1d749feb14d74`. Final compact evidence and local raw chains use `runs/gaussian-campaign-v01/nonlinear-bar-inverse-v02`.

The version-1 command `python -m experiments.run_nonlinear_bar_inverse run` intentionally reproduces the preserved moment-gate failure before sampling when given its frozen protocol. The version-2 command below requires the existing **LOCAL hash-bound observation archive**; a source-only fresh clone does not contain that NPZ, consistent with the user instruction to keep raw data local.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.continue_nonlinear_bar_inverse_v02 run \
  --protocol runs/operator-dev/nonlinear-bar-inverse-v02/protocol.json \
  --protocol-sha256 04e9baeb0116f2007536f83c3befb2dd920a71d7a48b2fc11fc1d749feb14d74 \
  --output /tmp/nonlinear-bar-inverse-fresh
```

Use a fresh output directory; the continuation requires the hash-bound local version-1 observation archive and preserved reference report. Five source tests cover prior-boundary/zero-force oracle parity, log-prior mapping, identifiability, likelihood channel counting, genuine finite-strain response, and numerical rejection gates. Three further analytic tests check centered Gaussian variance, invariance to a 250 mm constant shift, nonzero-center correction, and rejection of unresolved mass or negative variance. All posterior arrays, chain coordinates, and actual synthetic observations stay local.
