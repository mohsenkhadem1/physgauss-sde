# Nonlinear likelihood audit: measured results

The audit covers all 18 study configurations and all three proposal methods, using 256 independent proposals per method/configuration. It uses observed images only. The original study source is `d225760`; audit implementation is `6bc029d`. Core model modules were verified unchanged.

| Proposal | Configurations below relative ESS 0.1 | Median relative ESS | Minimum relative ESS |
| --- | ---: | ---: | ---: |
| physical | 8/18 | 0.1565 | 0.0080 |
| fixed_sde | 15/18 | 0.0442 | 0.0043 |
| adaptive_sde | 14/18 | 0.0339 | 0.0056 |

## What this means

- Several local posterior proposals have concentrated nonlinear likelihood weights. The local approximation is fragile in these configurations even when average per-pixel render disagreement appears small.
- These diagnostics qualify the adaptive-noise accuracy results; they do not invalidate the numerical linear-Gaussian evidence identity or demonstrate calibrated nonlinear uncertainty.
- Effective sample size here describes importance-weight concentration of independent proposals. It is not an MCMC mixing statistic and is not an independent experimental replicate count.
- The full nonlinear target can also be physically misspecified: a high ESS for the no-noise model on an unmodelled-force case would only show agreement with that wrong model's posterior.
- No corrected-posterior intervals are claimed, and no invalid physical proposal was silently replaced.

## Next inference milestone

Implement an iterative nonlinear smoother or a nonlinear posterior sampler, check convergence/mixing and model discrepancy, then repeat coverage and proper-score evaluation. Keep the zero-noise option and the matched process-noise controls. More scene diversity is required before any calibration/generalization claim.

## Reproduction

Use `physgauss audit --study runs/scale-study --source-ref d225760 --output runs/likelihood-audit --samples 256` for the historical raw local run. For a freshly generated current-version study use its committed source ref, typically `HEAD`. Raw posterior checkpoints remain in ignored run directories; compact audit outputs are in `examples/likelihood-audit/`. See `docs/LIKELIHOOD_AUDIT.md` for provenance and observation handling.
