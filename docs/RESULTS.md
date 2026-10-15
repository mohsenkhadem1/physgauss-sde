# First measured ablation

These are **two synthetic realizations of one known cube scene**, not a multi-scene benchmark. They share geometry, camera path, material parameters, observation-noise seed, observation schedule, and inference settings. Ground truth differs only in whether process forcing is active. Both methods use the same scene priors and known image noise. The SDE has additional latent increments and a separately simulator-trained diffusion prior.

The default scene has 21 timestamps, six 24x24 input images, and 64 posterior samples. The largest gap lies strictly between observed frames 6 and 14 (0.3 to 0.7 s); seven frames are withheld inside it. Only the four moving nodes contribute to position and uncertainty scores. The four clamped nodes are excluded.

## Largest-gap measurements

| Ground truth | Inference | Position RMSE (mm) | CRPS (mm) | 90% component coverage | Mean interval width (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| Stochastic | Physical, parameter uncertainty | 2.794 | 2.049 | 17.9% | 1.384 |
| Stochastic | Physical + neural SDE | 1.340 | 0.900 | 100.0% | 8.445 |
| Deterministic | Physical, parameter uncertainty | 1.221 | 0.654 | 72.6% | 2.508 |
| Deterministic | Physical + neural SDE | 1.244 | 0.864 | 100.0% | 8.749 |

The JSON snapshots are authoritative for unrounded values and the remaining scores. CRPS is computed for scalar position components; the report also includes a joint vector energy score and held-out/novel-view PSNR. These small known-appearance renders should not be compared with published real-scene PSNR tables.

## Interpretation

- When the truth contains process forcing, accounting for that forcing improves this realization's missing-motion error and CRPS.
- The uncertain deterministic model is too confident under that mismatch.
- The SDE's wide intervals cover all evaluated position components. This exceeds the nominal 90%; it is not evidence of calibrated uncertainty.
- When the truth is deterministic, adding the fixed calibrated diffusion prior does not improve position accuracy and produces unnecessarily broad intervals.
- The next experiment should infer or shrink process-noise amplitude, assess calibration over independent realizations, and test model mismatch. Do not tune an uncertainty temperature on these held-out results and call it independent calibration.

## Numerical and reproducibility checks

- All 12 local physical/numerical tests passed.
- A clean Linux GitHub Actions run passed the same tests and the complete quick demo.
- Default stochastic-run samples retained positive deformation determinants (minimum approximately 0.90).
- Internal net force and torque were at floating-point roundoff; clamped-node position error was zero.
- No invalid posterior sample was silently discarded.
- Checkpoint resume reproduced the stochastic experiment's scores and samples.
- Optimization records include gradient norms and Gauss-Newton decrements. The posterior is a local approximation around an approximate MAP; finite optimizer budgets and depth-sort boundaries remain limitations.

## Reproduce

```sh
python -m pip install -e .
physgauss check
physgauss demo --config configs/cpu_demo.json --output runs/cpu-demo
physgauss demo --config configs/cpu_deterministic.json --output runs/deterministic-control
```

Snapshots are in `examples/cpu-demo/` and `examples/deterministic-control/`. Exact source SHA-256 fingerprints, configurations, seeds, and dependency versions are in their `summary.json` files. Raw local runs additionally save checkpoints, input observations, and sampled trajectories. A resumed report's wall time excludes the previously cached optimization; use the per-method fit times to inspect that cost.
