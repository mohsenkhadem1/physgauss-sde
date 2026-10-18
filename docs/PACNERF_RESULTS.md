# PAC-NeRF external-data study: incomplete production run

**Status: three of sixteen production cases completed and independently replayed.**
One neural/constant pair is available; no overall or joint/marginal ranking is
available yet. The running
[GitHub Actions experiment](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39073611351)
uses source `b2d081a7a55ecf1027921f5d841c043e23713a9d`.
The matching CPU validation run 39073594746 passed all 102 numerical tests
and the historical smoke pipelines. Later reporting/audit edits do not change
the numerical model. See the [protocol](PACNERF_PROTOCOL.md) and
[incomplete report](../examples/pacnerf-sparse/report.html).

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
The two missing counterpart models are required for those comparisons.

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
