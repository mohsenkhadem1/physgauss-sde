# Nonlinear sampler: selected long-gap experiments

These are numerical inference experiments on three selected cases from the previous study, all with realization seed 3101 and the long-gap six-image schedule. They do not add independent scenes or provide a repeated-realization calibration study. The same known rest geometry, camera, density, supports, appearance and material family are used.

The sampler implementation is commit `b30fcff`; the historical local-reference study is commit `d225760`. Exact source hashes, checkpoint hashes and the numerical environment are preserved in the versioned JSON. Four chains each discard 1,000 warmup transitions and retain 2,000 transitions with gradient-guided pCNL plus independent and scale-transport moves. Each case therefore has 8,000 retained, correlated states. All states enter diagnostics; an evenly spaced 256-state rendering subset is used for predictive scores, matched by 256 independent local-reference draws.

## Convergence comes before prediction scores

The screen requires rank-normalized/folded split Rhat <1.01, bulk ESS >=400, tail ESS >=400, and no frozen chain for every monitored coordinate. Undefined or constant adaptive-scale diagnostics cannot pass automatically. Failing this screen leaves uncertainty estimates provisional. Passing cannot rule out missed modes or repair a wrong physical model.

| Truth | All monitored checks pass | Flagged quantities | Max defined Rhat | Min defined bulk ESS | Min defined tail ESS | Undefined quantities |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic | False | 4/190 | 1.138 | 19.116 | 185.518 | 1 |
| stochastic | False | 122/190 | 1.031 | 172.689 | 205.722 | 0 |
| unmodeled_pulse | False | 190/190 | 3.469 | 4.385 | 10.464 | 1 |

In the deterministic case, the four flagged quantities are stiffness, damping, initial vertical velocity and the constant zero-noise scale. The stochastic case's process-scale diagnostic passes, but the complete posterior does not. In the missing-force case, large between-chain disagreement is visible in the traces; predictive scores from those chains must not be interpreted as converged posterior estimates.

## Predictive scores, all withheld frames

These are single-realization scores from a finite rendering subset. Read them with the convergence screen; they are not evidence of calibrated coverage or superiority across scenes.

| Truth | Inference | Coordinate RMSE mm | CRPS mm | 90% coverage | Mean width mm |
| --- | --- | --- | --- | --- | --- |
| deterministic | local_reference | 0.457 | 0.261 | 94.4% | 1.638 |
| deterministic | nonlinear_mcmc | 0.484 | 0.278 | 98.9% | 1.722 |
| stochastic | local_reference | 1.734 | 0.968 | 90.0% | 5.979 |
| stochastic | nonlinear_mcmc | 1.722 | 0.950 | 90.6% | 5.512 |
| unmodeled_pulse | local_reference | 4.899 | 2.864 | 66.1% | 9.874 |
| unmodeled_pulse | nonlinear_mcmc | 5.155 | 3.235 | 44.4% | 6.693 |

## Component movement and physical rejections

- deterministic: 0/60 nontrivial scale-change proposals accepted; 0 invalid physical proposals in retained sampling. Per-chain scale occupations and warmup counters are recorded in JSON.
- stochastic: 20/298 nontrivial scale-change proposals accepted; 0 invalid physical proposals in retained sampling. Per-chain scale occupations and warmup counters are recorded in JSON.
- unmodeled_pulse: 30/1047 nontrivial scale-change proposals accepted; 0 invalid physical proposals in retained sampling. Per-chain scale occupations and warmup counters are recorded in JSON.

## Interpretation and next steps

The new Metropolis kernel has the full nonlinear likelihood as its target and passes tests against numerical quadrature, including nonlinear scale probabilities. That mathematical correction does not establish finite-run convergence. Check the table and detailed quantity diagnostics before using intervals. A constant scale trace can reflect concentrated probability or poor component exploration; the current screen deliberately does not resolve that ambiguity in its favour.

The unmodelled external pulse remains a misspecified inference model: better posterior sampling cannot create the missing physical force law. The next steps are stronger component exploration, convergence-checked fixed-scale controls where useful, and an explicit structured force/discrepancy model. Only then should coverage be evaluated over new realizations and geometries, followed by uncertain camera/geometry and real video.

See [the derivation and exact command options](NONLINEAR_SAMPLING.md) and [the self-contained trace/report artifact](../examples/nonlinear-sampling/report.html). The raw `chains.pt` tensors remain in ignored local run directories. Compact CSV, JSON and PNG measurements are committed for handover.

## Reproduce the selected-case plan

From a clean checkout, first generate local references using the existing full study configuration, then sample the three selected cases. Use the recorded numerical environment for exact realization-level comparisons; cross-platform eigensolver/RNG differences can change the synthetic realization.

```sh
physgauss study --plan configs/scale_study.json --output runs/new-scale-study
physgauss sample --study runs/new-scale-study --source-ref HEAD --case deterministic__long_gap__3101 --case stochastic__long_gap__3101 --case unmodeled_pulse__long_gap__3101 --output runs/new-nonlinear-long-gap --chains 4 --warmup 1000 --draws 2000 --evaluation-samples 256 --langevin
```

The executed local investigation used `--study runs/scale-study --source-ref d225760`, with all other settings as above. That historical raw study is not redistributed as large checkpoints; its compact measurements and source commit are preserved. The fresh-study route is the handover path for a new clone. These runs resume only at completed-case boundaries with matching settings and source; an interrupted case restarts from its recorded seed.
