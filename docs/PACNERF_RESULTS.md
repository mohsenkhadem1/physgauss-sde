# PAC-NeRF external-data study: incomplete production run

**Status: one of sixteen production cases completed and independently replayed.**
No neural/constant or joint/marginal ranking is available yet. The running
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

Independent NumPy replay reproduces posterior means/intervals to below 1e-12 m,
gap scores and the subspace lower bound, and sampled cached exact-target ratios
to 1.9e-10. Force total covariance is checked against E[S]+Cov[m]. These checks
support scoring and numerical provenance, not model truth or force calibration.

## Physical and data audits

Supplied reference particles project onto the corresponding official images
at frames 0,3,6,13 for both objects. This visual audit supports the coordinate
and time conventions; it is not a fitted reconstruction.

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
