This checks how the unchanged nonlinear bar inverse behaves under repeated measurement noise. It uses **64 fresh, independent noise realizations at the same fixed truth**: material stiffness E=6000 Pa and axial spring stiffness K=220 N/m. It measures conditional frequentist coverage at that truth, rather than Bayesian simulation-based calibration or calibration of real physics.

The prior remains uniform in log E over 4500–9000 Pa and log K over 100–400 N/m. The support location and spring law are known. Each realization has four independent 200 µm Gaussian errors on left/right displacements under known 1 N and 3 N loads. The held 2 N load tests static force interpolation. The previously reported single noise realization, which missed both parameter intervals, remains preserved and is excluded from these prospective 64 draws.

| Quantity | Covered by latent 90% interval | Exact marginal 95% binomial CI | Mean interval width | Posterior mean bias | Posterior mean RMSE |
|---|---:|---:|---:|---:|---:|
| E | 54/64 (84.4%) | 73.1–92.2% | 389.6 Pa | −15.2 Pa | 128.0 Pa |
| K | 55/64 (85.9%) | 75.0–93.4% | 10.10 N/m | +0.258 N/m | 3.44 N/m |
| Held left displacement | 55/64 (85.9%) | 75.0–93.4% | 0.416 mm | −0.0067 mm | 0.142 mm |
| Held right displacement | 56/64 (87.5%) | 76.8–94.4% | 0.407 mm | +0.0233 mm | 0.128 mm |

Each confidence interval includes 90%, with substantial sampling uncertainty. There was no requirement to obtain a particular coverage count. K and left-displacement coverage are identical because the held left displacement is exactly 2/K. The confidence intervals in the table are marginal, not a joint guarantee across quantities. These are latent displacement intervals; they do not include a new target measurement's noise.

All 64 posteriors passed the declared numerical checks. With c=1/K, the conditional density is a Gaussian in c multiplied by the correct 1/c prior. A bounded series integrates this conditional, leaving continuous adaptive integration in log E. Both outer tolerances gave identical coverage classifications and interval endpoints at the declared bisection resolution; the largest truth-CDF difference was 7.01×10⁻¹⁴. Eight fixed indices were independently checked against two-dimensional integration of the original four-channel likelihood: maximum CDF difference 1.17×10⁻⁹ and relative normalizer difference 7.29×10⁻¹³. Centered moments avoid the earlier raw-moment cancellation problem.

The terminal receipt explicitly corrects the series bound's interpretation: the implemented geometric bound is relative to the expansion center, so multiplying by a conservative factor two bounds relative integral error for all inverse powers used. The resulting bound is 5.32×10⁻¹⁷ across the entire E prior and all 64 observations, below the unchanged 10⁻¹⁴ criterion. No source, observation, likelihood, or result was changed after the run. Adaptive integration error estimates are numerical diagnostics, not rigorous certificates.

The run took 61.35 seconds with two CPU threads and a predeclared 300-second cap. All cases were retained; there were no retries, new finite-element solves, or sampler runs. The truth lies at approximately 0.415 and 0.569 in the two normalized log-prior coordinates. Near-boundary truths were not tested. This homogeneous static bar has no uncertain support location, model discrepancy, contact, Isaac observations, or population of geometries. The result supports numerical reliability of this small inverse; it does not establish calibration for those broader problems.

Source: [conditional integration](../experiments/campaign_bar_noise_population.py), [runner](../experiments/run_bar_noise_population.py), and [four synthetic checks](../tests/test_campaign_bar_noise_population.py). The [prospective protocol](../runs/operator-dev/nonlinear-bar-noise-population-v01/protocol.json), [compact results](../runs/operator-dev/nonlinear-bar-noise-population-v01/result-summary.json), and [terminal receipt](../runs/operator-dev/nonlinear-bar-noise-population-v01/terminal-receipt.json) bind the completed run. Actual noise arrays remain local. The preceding [single-realization inverse](GAUSSIAN_NONLINEAR_BAR_INVERSE.md) retains its interval misses and failed original moment gate.

Reproduction requires the recorded source versions. Use a fresh protocol and output path; the original paths refuse overwrite:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m experiments.run_bar_noise_population \
  --prepare runs/operator-dev/nonlinear-bar-noise-reproduction/protocol.json
# Pass the printed SHA256 and the new protocol path to --protocol / --sha256,
# with a fresh --output directory. This regenerates the predeclared noise seed.
```
