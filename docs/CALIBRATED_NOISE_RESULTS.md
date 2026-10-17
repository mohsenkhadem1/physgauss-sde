# Calibrated noise: final sparse-view results (v0.8.1)

Noise-amplitude calibration improved substantially, but the neural model still
does not beat constant internal noise for reconstruction. Whole-gap prediction
sets passed the held-out coverage check with wider intervals. **The nonlinear
Bayesian posterior remains unvalidated: every monitored quantity fails the
finite-chain convergence screen.** These are different findings.

Open `examples/calibrated-noise/report.html`. Read `CALIBRATED_NOISE_METHOD.md`
and `CALIBRATED_NOISE_PROTOCOL.md` for equations, independent data splits,
corrections, and reproduction commands. Historical reports are unchanged.

## What was tested

The final experiment has 114 fits, all using six RGB images from one moving
camera, the same known scene, and exact conditional Gaussian force elimination:

- 36 matched fits: four truth conditions × three original seeds × three noise
  models (independent floor, calibrated constant, calibrated neural).
- 38 interval-calibration fits: 19 new motions × constant/neural models.
- 40 held-out fits: 20 fresh motions × constant/neural models.

All 114 pass the predeclared optimizer stationarity threshold. The paired
benchmark's exact RGB and original accurate reference states are preserved.
The two stochastic conditions share three underlying seeds; they are not six
independent realizations. Averages below are means of per-case coordinate RMSE.

## Noise calibration

The network retains 33 learned parameters but now uses scaled smooth magnitudes
of strain/speed, independent simulated operating trajectories, and replicated
innovation observations. Weights freeze before all reconstructions. The
constant control is estimated directly from training innovations.

- Four independent calibration-validation motions: amplitude RMSE **0.281%**,
  standardized innovation variance **1.009**, nominal-90% innovation coverage
  **89.85%**.
- Twenty fresh final test motions: pooled frame-state amplitude RMSE **0.295%**.
- The previous v0.7 audit found **36.3%** amplitude error in the benchmark's
  operating regime. The final report also compares old/new networks on exactly
  those same benchmark states.

This is independent synthetic simulator supervision, not diffusion learned
from a sparse test video. Calibration states within a trajectory are correlated;
fixed/support edges are included in the amplitude diagnostic. Network-weight
uncertainty is not propagated.

## Reconstruction under the same original images

Mean missing-gap coordinate error pooled across the two stochastic conditions:

| Model | Gap error (mm) | Local weight screens passed |
|---|---:|---:|
| Independent floor | 4.847 | 12/12 |
| Calibrated constant internal noise | 3.281 | 8/12 |
| Calibrated neural internal noise | 3.623 | 0/12 |

The previous v0.7 constant/neural means were 2.893/2.974 mm. More accurate noise
amplitudes did not improve these point reconstructions. The final pipeline also
includes the numerical corrections below, so the old/new comparison does not
isolate calibration alone. Constant versus neural **within the final run** is
the controlled amplitude-model comparison. No neural-specific gain is shown.

On the 20 fresh stochastic-mixture test motions, mean errors are **3.522 mm**
for constant and **4.380 mm** for neural noise. All cases are retained.

## Prediction-set uncertainty

Each calibration motion supplies one maximum standardized error over all 84
free-node xyz components in the seven-frame missing interval. The frozen
multiplier is the 18th of 19 scores, targeting 90% simultaneous whole-gap
coverage. Tests use seeds 83001–83020, never used for tuning.

| Model | Raw component 90% coverage | Frozen SD multiplier | Whole-gap test coverage | 95% Wilson interval | Mean whole-gap-set component width |
|---|---:|---:|---:|---:|---:|
| Constant | 91.37% | 2.931 | 19/20 (95%) | 76.4–99.1% | 21.70 mm |
| Neural | 81.13% | 3.621 | 20/20 (100%) | 83.9–100% | 26.28 mm |

The unscaled local component intervals have mean widths 12.18/11.94 mm.
Whole-gap prediction sets target a stricter event than coordinate-wise bands;
these coverage rates must not be interpreted as the same event. The neural
sets are wider and do not establish a better uncertainty/accuracy tradeoff.
Twenty tests leave broad statistical uncertainty; there is no demonstrated
coverage advantage over constant noise.

Split-conformal prediction controls marginal repeated-case coverage under
exchangeability, averaging over calibration-set randomness. It does not give
a guarantee for every video, another scene/view schedule, or distribution shift.
It does not correct or validate the Bayesian force/material posterior.

## Nonlinear posterior audit: failed exploration

The predeclared neural pulse-plus-stochastic seed-4101 case uses a refined
mode, exact observed-curvature proposal, four independent pCN chains, 1,000
warmup and 2,000 retained transitions per chain. The full nonlinear
marginalized-force target determines Metropolis acceptance. All 1,448 state
coordinates, 84 gap components, target density and 15 conditional force means
are monitored (1,548 quantities).

- **1,548/1,548 fail** the complete screen: rank/folded split Rhat <1.01,
  bulk/tail ESS >=400, and no frozen chains.
- Worst Rhat: **1.302**; minimum bulk/tail ESS: **10.5 / 18.3**.
- Fisher-Gaussian relative importance ESS: **0.0138**; observed-Hessian
  Gaussian: **0.0706**, both below the 0.1 warning threshold (512 proposals).
- Local curvature itself checks correctly: maximum full-target AD directional
  relative error about **5.3e-12** and refined finite-difference error **3.5e-5**.

Correct curvature and a correct Metropolis kernel are insufficient evidence of
adequate finite-chain exploration. The saved MCMC position/force summaries are
diagnostics, not validated posterior estimates or a replacement model ranking.
Force conditionals are exact at a fixed trajectory; mixing them over an
unvalidated trajectory distribution does not make the complete force posterior
reliable. Position prediction-set coverage above remains a separate result.

## Corrections and failed initial execution

Cross-machine reconstruction of an initial CI case changed its normalized
objective from 1529.776 to 1554.760. The cube RBF kernel has a triple eigenvalue
at the retained-rank boundary: different backends selected different spatial
subspaces. The final code freezes the projection already published in v0.7,
including it in the source fingerprint. Simulation and inference now share it
on every backend. It remains a declared rank-three approximation.

A second check exposed incorrect emission derivatives in exactly empty depth
slabs. The renderer now evaluates the analytic limit of
`(1-exp(-tau))/tau` at zero. Tests check both first and second derivatives
against the closed-form volume solution. The analytic forward model and exact
original observed images are retained.

The initial 36/78-fit summaries and reproduction failure are retained under
`examples/calibrated-noise-diagnostics/`; the initially reported 19/20 neural
coverage is superseded. No model or noise hyperparameter was tuned on those
test outcomes. The final held-out seeds are fresh. Every final benchmark and
validation fit uses both fixes.

## Provenance and handover

- Numerical implementation: `edbf12a4a872de1fdd1f7be4bd5ba687a7fe50c6`,
  source fingerprint
  `7a4e3ce81327156803e359031e7365f001c275607b7f8026c997078b33769f96`.
- [All 36 matched fits](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39045544526)
  and [all 78 independent-validation fits](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39044647259)
  succeeded in recorded CPU partitions, with calibration frozen before testing.
- Every saved normalized target, solver state, uncertainty factor and interval
  score is independently reconstructed locally without optimization. The source,
  weights, exact RGB and serialized prior initialization are checked.
- The publication contains exact observed pixels, reference states/audits,
  every solver coordinate, network weights, frozen quantiles, force moments,
  per-case CSV, all failed inference flags, and source/artifact hashes.
- Raw primary inputs/checkpoints are in `runs/calibrated-noise-published`;
  posterior chains are in `runs/calibrated-posterior-optimized`. Large precision
  matrices/chains remain local or regenerate from code, inputs and seeds.

This is a known eight-node/six-tetrahedron scene with fixed cameras, appearance,
density and supports, one view schedule and restricted force location/profile.
There is no real-video, unknown-geometry/camera, mesh-convergence or
out-of-distribution uncertainty validation. The positive result is improved
synthetic noise calibration and useful held-out position prediction sets;
Bayesian posterior validation and a neural reconstruction advantage remain open.
