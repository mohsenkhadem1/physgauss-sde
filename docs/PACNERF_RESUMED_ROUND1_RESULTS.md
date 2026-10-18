# Corrected PAC-NeRF sparse-view comparison

Incomplete or failed validation: no overall scientific ranking.

16/16 final worker summaries; 10/16 finite-chain passes.

[Actions experiment](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39100239747) · [Offline report](https://github.com/mohsenkhadem1/physgauss-sde/blob/main/examples/pacnerf-resumed-round1/report.html) · [Frozen protocol](https://github.com/mohsenkhadem1/physgauss-sde/blob/main/docs/PACNERF_EXACT_PROTOCOL.md)

Frozen worker `219665b`; exact linear integrals, exact causal innovation coordinates, fixed full-RGB Gauss-Newton proposal metric, full nonlinear Metropolis target.

| Case | Eligible finite chains | Gap coordinate RMSE (mm) | 90% component coverage |
|---|---:|---:|---:|
| torus__4views__constant__marginal | True | 158.101 | 12.41% |
| torus__4views__constant__joint | True | 158.192 | 12.20% |
| torus__4views__neural__marginal | True | 158.351 | 12.46% |
| torus__4views__neural__joint | True | 158.225 | 12.59% |
| torus__6views__constant__marginal | True | 97.475 | 17.45% |
| torus__6views__constant__joint | True | 97.444 | 18.03% |
| torus__6views__neural__marginal | True | 97.507 | 17.84% |
| torus__6views__neural__joint | True | 97.396 | 17.64% |
| bird__4views__constant__marginal | True | 112.108 | 50.35% |
| bird__4views__constant__joint | False | 112.161 | 50.91% |
| bird__4views__neural__marginal | True | 112.068 | 49.31% |
| bird__4views__neural__joint | False | 111.935 | 50.48% |
| bird__6views__constant__marginal | False | 88.479 | 5.86% |
| bird__6views__constant__joint | False | 87.524 | 8.20% |
| bird__6views__neural__marginal | False | 87.092 | 15.17% |
| bird__6views__neural__joint | False | 86.811 | 15.49% |

Two deterministic official elastic simulations, Torus and Bird. Initial geometry and eleven-camera initial appearance are supplied; 4/6 frames mean 3/5 later images. Camera switching is not continuous moving-camera capture. The model is a twelve-mode elastic graph with frame-held contact and 32px Gaussian rendering, not the original MPM.

Marginal and joint loads have the same integrated motion posterior. Differences beyond Monte Carlo uncertainty invalidate equivalence claims. Force coefficients are a restricted acceleration field; no force ground truth is supplied. Coverage is descriptive, not repeated-dataset calibration.

Failed or unfinished cells are excluded from quality rankings. The report preserves their status and any failed-chain scores as diagnostics. Gradient-batches/pose-ESS are a compute proxy, not a hardware-normalized runtime claim. Checkpoint counters exclude unsaved work after interrupted attempts.

Compact reports are committed; raw states and inputs remain in Actions artifacts and local copies. Artifact retention is 30 days.

This snapshot combines the original nine completed arms with seven explicitly resumed runs. See continuation-integrity.json for source runs, exact prefix and RNG checks. The original report and archive remain unchanged.
