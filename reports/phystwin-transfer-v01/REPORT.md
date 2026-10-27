# Controlled PhysTwin transfer and probing reference

2026-10-27. Completed controlled physics study, not an end-to-end visual reconstruction or calibrated active-learning system.

## Design and numerical checks

A 240 × 80 × 80 mm block has compliant attachments on both end faces. Two physical explanations are deliberately ambiguous from one end: soft material (2 kPa) with firm support (1000 N/m), and stiff material (8 kPa) with loose support (64 N/m). Both use a 1000 N/m moving actuator. An initial 9 mm axial displacement produces end observations differing by just 0.071 mm, while the complete displacement fields differ by 2.129 mm coordinate RMS.

Independent compressible Neo-Hookean FEM supplies quasistatic reference trajectories. All six object/action combinations passed the frozen nested-mesh gate: decreasing differences and fine-grid common-point RMS below 0.3 mm. Actual fine-grid differences were 0.0016–0.0157 mm. This establishes a refinement indicator, not an exact continuum error bound. No gravity or contact is included here.

PhysTwin uses the official dense optimizer and spring simulator at revision `81c718790a37e5e0102eb77af2c6edd34a9db25f`, with known rest geometry, physical lumped masses and an explicit gravity-off adapter. We fit 80 iterations using only the nine right-end points during the initial interaction. There is no CMA topology search, raw image inference, Gaussian appearance fitting, or uncertainty learning in this adaptation. Held interactions receive prescribed controller trajectories and reset initial states; held object positions enter scoring only.

The 61-frame trajectory consists of a three-second ramp and three-second hold. Scores use the last ten frames, comparing dynamic relaxation against static FEM. Ramp frames in videos interpolate quasistatic FEM solutions; these are not validated dynamic ground truth. Halving the simulator timestep changed final-hold coordinates by at most 0.000652 mm RMS. All held prediction gates remain passed. Nine unit tests passed, including FEM energy/tangent derivatives, force balance and posterior utility limiting cases.

## Prediction results

Coordinate RMSE in mm over the complete 81-point field (not average Euclidean point distance):

| Object | Initial axial (fit interaction) | Held sideways | Held oblique |
|---|---:|---:|---:|
| Soft / firm support | 0.279 | 0.243 | 0.263 |
| Stiff / loose support | 1.822 | 0.406 | 0.920 |

All four held scores beat the frozen 1.5 mm threshold and stationary baseline (3.21–3.86 mm). However, the stiff/loose initial-interaction fit has only 0.000988 mm error at the observed end while its complete-field error is 1.822 mm. Visible fit alone does not establish correct hidden deformation or physical parameters. This is two controlled objects, not a population benchmark.

Videos remain local under `runs/phystwin-transfer-v01/{soft_firm,stiff_loose}/{axial,sideways,oblique}.mp4`. They show FEM reference versus PhysTwin, at fixed scale with no deformation amplification. They are point-cloud diagnostics, not photorealistic reconstructions.

## Separate finite-hypothesis probing reference

After the transfer gate passed, a separate numerical posterior was tested over nine equally weighted E/support combinations. Its n4 FEM forward bank is compared against independent n8 truth. This posterior is not derived from PhysTwin springs. Observations have independent Gaussian noise with 0.3 mm standard deviation. Each of the two truth objects has 32 noise repetitions, for 64 trials; these are not 64 independent objects.

The initial observation contains 27 axial end-point coordinates. A probe observes three coordinates of a previously hidden midpoint marker after a reset axial or sideways interaction. Thus the experiment assumes new marker/view access as well as another interaction; the gain cannot be attributed solely to force selection. The prediction target is the complete displacement field under the held oblique interaction.

The fixed action minimizes expected target variance under the prior, before outcomes. Adaptive selection minimizes expected variance after the initial observation. Random results exactly average both reset actions with matched observation budgets. Gauss-Hermite orders 7 and 11 agree on all 64 adaptive choices (maximum risk difference 3.27e-10 m²); this supports choice stability, not convergence of an arbitrary continuous material posterior.

| Policy | Target RMSE (mm) | Nominal 90% coordinate coverage | Mean interval width (mm) |
|---|---:|---:|---:|
| Initial observations only | 0.468 | 66.8% | 0.768 |
| Adaptive probe | 0.080 | 44.6% | 0.127 |
| Prior-optimal fixed probe | 0.080 | 44.6% | 0.127 |
| Random probe | 0.188 | 54.5% | 0.264 |

Adaptive selected axial on every trial, exactly matching fixed. There is no demonstrated adaptive advantage. Point predictions improve substantially with the new observation, but intervals narrow and coverage worsens. Discrete hypothesis support and forward-model discrepancy are plausible contributors; this run does not isolate their effects. Coordinate coverage on two fixed objects is not population calibration.

## Decision

The physics prediction adapter works on this narrow held-interaction test. Trustworthy uncertainty and a useful adaptive policy remain unresolved. Next establish a continuous, refined material/support posterior with an explicit discrepancy model and held-out calibration objects. Expand candidate load directions and sensor locations only with a frozen test design that admits object-dependent optimal actions. Compare against the same strong fixed policy. Do not claim physical recovery or adaptive superiority from this study.

## Reproduction

Run from the repository root. `.venv` must contain the repository FEM/test dependencies; the separate PhysTwin environment is installed by the setup script. Raw banks, checkpoints and videos are intentionally kept local.

```bash
bash scripts/phystwin/setup.sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/phystwin/controlled_reference.py --output runs/phystwin-transfer-v01/reference
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 runs/phystwin-env/bin/python scripts/phystwin/run_controlled.py --object soft_firm
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 runs/phystwin-env/bin/python scripts/phystwin/run_controlled.py --object stiff_loose
CUDA_VISIBLE_DEVICES=0 runs/phystwin-env/bin/python scripts/phystwin/run_controlled.py --object soft_firm --evaluate-only --step-refinement 2
CUDA_VISIBLE_DEVICES=1 runs/phystwin-env/bin/python scripts/phystwin/run_controlled.py --object stiff_loose --evaluate-only --step-refinement 2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/phystwin/controlled_probing.py
.venv/bin/python -m pytest -q tests/test_phystwin_controlled_reference.py tests/test_phystwin_controlled_probing.py tests/test_phystwin_metrics.py
```

The frozen designs are `protocol.json` and `probing-protocol.json` alongside this report. Numerical summaries, ambiguity measurement, timestep comparison and test output are included. The probing script reuses its local response bank; remove that bank before changing its design or solver.
