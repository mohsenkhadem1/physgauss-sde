# PAC-NeRF external-data study: superseded quadrature experiment

**Numerical target defect confirmed; corrected rerun pending.** The completed
cells below describe the historical approximate-integrator target. They are not
results for the corrected model, and no overall scientific ranking is claimed.

**Status: six of sixteen production cases completed and independently replayed;
five pass finite-chain diagnostics, one fails.**
The entire six-view Torus four-arm regime is available under that historical
target. Ten original jobs reached the time cap; their continuations were stopped
after the integration failure was confirmed. The original
[GitHub Actions experiment](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39073611351)
uses source `b2d081a7a55ecf1027921f5d841c043e23713a9d`.
The matching CPU validation run 39073594746 passed all 102 numerical tests
and the historical smoke pipelines. Later reporting/audit edits do not change
the numerical model. See the [protocol](PACNERF_PROTOCOL.md) and
[incomplete report](../examples/pacnerf-sparse/report.html).

## Integration failure and exact-integral correction

Sixteen probes per saved case exposed a failure missed by the low-stiffness
Torus checks. The Bird four-view constant/marginal prefix has standardized
stiffness near 9.15. There the four-point covariance quadrature has maximum
relative covariance eigenvalue error **0.90885**, and relative constant-load-map
error **0.009249**. These are numerical integration errors at the checked states,
not reconstruction errors or a causal explanation of all inference failures.

Independent 128- and 256-point quadrature agree to 2.47e-14 in normalized
covariance; the full-matrix reference agrees with them to 2.56e-12. This confirms
the failure is in the four-point approximation, rather than the reference.
The full historical target is therefore superseded, including cells whose
finite-chain checks passed. Old raw inputs, chains, summaries and failed or
cancelled attempts remain preserved; they must not be mixed with corrected chains.

The exact prototype balances each damped mode, integrates coupled covariance
blocks with stable 5x5 augmented Kronecker-sum exponentials, and integrates
constant loads with 3x3 augmented exponentials. Four independent analytic,
full-matrix, gradient and Hessian controls pass. At all **256 saved-state probes**,
maximum relative covariance/load errors are 2.57e-12 / 6.65e-13. These are
prototype validation results, not a corrected sparse-view comparison. The
mechanics, priors, observations, noise weights and transferred neural network
have not been retuned. New posterior runs are required after integration is
corrected and the numerical target is frozen again.

## First completed case: Torus, four images, neural noise, marginalized loads

Four chains each retain 2000 draws after 500 discarded warmup transitions.
All 353 monitored scalars pass the finite-chain gates: worst Rhat 1.00302,
minimum bulk/tail ESS 1589/964. This does not prove global exploration.

| Quantity | Result |
|---|---:|
| Gap particle-coordinate RMSE, posterior mean | 81.48 mm |
| Gap particle-coordinate RMSE, MAP | 80.04 mm |
| Evaluation-only best possible error in fixed shape subspace | 18.36 mm |
| Component coverage of nominal 90% intervals | 13.32% |
| Mean 90% interval width | 34.17 mm |
| All held-out image PSNR | 25.79 dB |
| Background-difference-region PSNR | 16.21 dB |

This is poor external-data transfer and severe interval undercoverage. There is
no evidence yet that neural noise helps, or that marginalization is faster.
Valid counterpart posteriors are required for those comparisons.

The second completed case is **Torus, six views, neural noise, marginalized
loads**: gap coordinate RMSE 97.54 mm, nominal 90% component coverage 18.10%,
and mean width 31.19 mm. All 353 finite diagnostics pass (worst Rhat 1.00422,
minimum bulk/tail ESS 922/590), with 2000 retained draws per chain. Independent
raw replay also passes. Its fixed-subspace oracle floor is 14.87 mm. The report
now shows its actual withheld images against the rendered posterior mean pose.
The 4/6 schedules differ in cameras and gap length, so these two numbers do not
isolate the effect of adding views. Both of these cases use neural noise with
marginalized loads.

## First matched noise comparison: Torus, six-view schedule, marginalized loads

| Quantity | Constant noise | Neural noise |
|---|---:|---:|
| Gap coordinate RMSE | 97.46 mm | 97.54 mm |
| Nominal 90% component coverage | 17.84% | 18.10% |
| Mean 90% width | 31.15 mm | 31.19 mm |
| Background-difference-region PSNR | 15.6343 dB | 15.6344 dB |
| Retained draws per chain | 4000 | 2000 |
| Worst monitored Rhat | 1.00335 | 1.00422 |
| Minimum bulk / tail ESS | 1332 / 969 | 922 / 590 |

Both pass finite sampling diagnostics and independent raw-state/force replay.
Constant noise required its declared extension to 4000 draws per chain. Their
reconstructions and interval widths are practically similar in this one regime,
with no demonstrated accuracy or calibration benefit from the transferred
neural amplitude. This is one object and schedule, not a general ranking or
a hardware speed comparison. Their actual withheld images are shown side by
side in the report. Explicit-force counterparts and the remaining regimes are
still running.

## First complete four-arm regime: six-view Torus

| Variant | Gap coordinate RMSE | Nominal 90% coverage | Draws per chain | Gradient batches / worst common pose ESS |
|---|---:|---:|---:|---:|
| Constant, marginalized loads | 97.46 mm | 17.84% | 4000 | 22.38 |
| Constant, explicit joint loads | 97.41 mm | 17.64% | 4000 | 51.03 |
| Neural, marginalized loads | 97.54 mm | 18.10% | 2000 | 19.67 |
| Neural, explicit joint loads | 97.48 mm | 17.90% | 4000 | 46.40 |

All four pass finite diagnostics and independent raw-score, cached-target and
full force-moment replay. The joint cases' worst Rhat is 1.00857 (constant) and
1.00247 (neural), with minimum bulk/tail ESS 735/686 and 705/592. Their saved
retained invalid/divergent proposal counts remain available; proposals are
Metropolis-corrected and invalid trajectories reject. Finite checks do not prove
global exploration or remove all numerical/exploration concerns.

All four aggregate reconstruction errors differ by under 0.13 mm, while all
severely undercover. Marginalization has not improved physical accuracy here,
as expected for equivalent integrated posteriors. It uses about **2.28 times
(constant) and 2.36 times (neural) less gradient work per worst common modal-pose
ESS** in this regime. This is one observed computational comparison, includes
warmup gradient batches but excludes mode fitting, and is not a universal
hardware speed claim. The explicit runs each use about 77k gradient batches;
the marginalized runs use 40.9k and 24.3k.

The constant pair's largest modal-mean discrepancy is 2.37 combined batch-MCSE
units. The neural pair reaches 3.64, with two coordinates just beyond the
approximate within-pair 95% Bonferroni cutoff of 3.60. This borderline check
deserves recording rather than declaring exact finite-sample agreement. The
predeclared full eight-pair screen and all other result gates remain incomplete;
similar aggregate errors alone do not validate the full posterior or forces.

A post-hoc covariance diagnostic helps interpret the similar noise results.
At 256 stratified saved states per six-view Torus marginalized arm, the fixed diffusion
floor contributes a median 94.0–94.1% of the neural model's **one-step particle
position process variance**. Evaluating both noise models at identical paths,
the median neural/constant variance ratio is only 1.0075–1.0078, although the
95th percentile is about 1.30. This describes a local conditional covariance;
it excludes uncertain loads, later propagation and the posterior distribution.
It is not an attribution of reconstruction error or a full uncertainty
decomposition. The floor and neural weights remain unchanged.

The force figure shows each available arm's unknown acceleration averaged over
handles, with posterior mean plus/minus one standard deviation. The gray region
is the unobserved gap. Inferred loads may compensate for reduced mechanics or
image-model mismatch; neither their magnitude nor their uncertainty is physically
validated by this dataset. They are not measured contact forces or Newton values.

Independent NumPy replay reproduces posterior means/intervals to below 4e-12 m,
gap scores and the subspace lower bound, and sampled cached exact-target ratios
to 2.4e-10. Force total covariance is checked against E[S]+Cov[m], and all
retained-state conditional force moments are replayed. These checks
support scoring and numerical provenance, not model truth or force calibration.

## Late inference failure: Torus, four views, constant noise, marginalized loads

This case reaches the declared 8000 retained draws per chain but fails badly:
worst Rhat 1.3783, minimum bulk/tail ESS 8.73/76.54. An interim 1000-draw check
passed; 2000, 4000 and 8000 fail. Material-parameter traces look mixed while
motion/velocity traces reveal late switches between distinct solutions, and
the chains have not agreed by the end. The trace audit uses no future geometry.

Its numerical scores and full force moments replay correctly, but **they are
not valid converged posterior estimates** and are excluded from model rankings.
The four-view neural/constant comparison is therefore unresolved. Passing a
short finite diagnostic elsewhere does not prove all remote solutions were seen.

The replay audit also caught a verifier-only draw-selection assumption: production
uses default float32 Torch linspace before selecting at most 2000 interval draws.
At 32000 total retained draws, float64 NumPy linspace selected two different
indices. Matching the production rule restores quantile replay to 1.5e-12 m.
No production intervals, model, chains or scores were changed by this correction.

A separate [causal-innovation algebra audit](PACNERF_INNOVATIONS.md) tests an
equivalent coordinate system as a possible future inference improvement.
It reproduces the original density, including the Jacobian, on sixteen saved
states per completed case to at most 1.23e-11. This is not a new sampling result:
all production chains and reported convergence failures remain unchanged.

## Physical and data audits

Supplied reference particles project onto the corresponding official images
at frames 0,3,6,13 for both objects. This visual audit supports the coordinate
and time conventions; it is not a fitted reconstruction.

An evaluation-only render audit holds the original colors, Gaussian radius,
opacity, 32px resolution and depth integration fixed, but supplies either the
true future particle centers or their least-squares modal projection. Over the
six-view schedule's 138 held-out image pairs:

| Region PSNR | True centers supplied | Projected geometry supplied |
|---|---:|---:|
| Torus | 18.15 dB | 17.75 dB |
| Bird | 19.35 dB | 19.47 dB |

Visible rendering error remains even with true geometry. These are post-hoc
diagnostic references, not sparse reconstructions or upper bounds on RGB PSNR;
inaccurate geometry can compensate for image-model errors. Production inference
is unchanged. Background compositing also lacks table-depth occlusion.

The linear propagator/noise quadrature passes independent full-matrix checks
(worst relative covariance error 2.85e-7). However a post-hoc nominal, zero-noise,
zero-load check finds 16.41/17.20 mm whole-sequence coordinate differences for
Torus/Bird between frame-held contact and a refined solve of the same reduced
ODE. The refined references agree below 3e-13 m. Contact integration is therefore
a material approximation. These numbers do not decompose the 81.48 mm posterior
error, and do not validate the original MPM. The production model is unchanged.

Other limits include twelve infinitesimal elastic modes, 32px rendering,
restricted load profiles, and noise transferred from the earlier cube study.
Initial geometry and all eleven initial camera images are supplied; the later
sparse observations switch among fixed cameras. This is a conditional transfer
benchmark, not unknown-geometry monocular reconstruction or full PAC-NeRF.

Compact results, source/data hashes, artifact identifiers and the current workflow
snapshot are versioned with the report. Full raw chains remain Actions artifacts
with 30-day retention. The collector and independent verifier commands are in
the protocol so work can resume without refitting completed cases.
