# Adaptive noise study: first 18-case results

Three independent seeds per condition/schedule, one known geometry, six input images per schedule. All numbers below score **every withheld timestamp**, unlike the milestone-1 largest-gap table. All controls share a common local image likelihood. Source implementation: `d225760`.

| Truth | Schedule | Method | RMSE (mm) | CRPS (mm) | 90% coverage | Mean width (mm) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| deterministic | long_gap | Adaptive mixture | 0.658 | 0.388 | 75.7% | 1.634 |
| deterministic | long_gap | Fixed SDE | 1.756 | 1.024 | 96.1% | 7.780 |
| deterministic | long_gap | No process noise | 0.670 | 0.385 | 77.2% | 1.748 |
| deterministic | short_gap | Adaptive mixture | 0.783 | 0.452 | 82.4% | 2.169 |
| deterministic | short_gap | Fixed SDE | 1.474 | 0.870 | 99.1% | 7.184 |
| deterministic | short_gap | No process noise | 0.851 | 0.485 | 80.9% | 2.164 |
| stochastic | long_gap | Adaptive mixture | 2.004 | 1.123 | 83.1% | 5.356 |
| stochastic | long_gap | Fixed SDE | 2.373 | 1.316 | 91.7% | 7.956 |
| stochastic | long_gap | No process noise | 3.132 | 2.232 | 20.4% | 1.334 |
| stochastic | short_gap | Adaptive mixture | 1.788 | 0.998 | 88.7% | 5.808 |
| stochastic | short_gap | Fixed SDE | 1.899 | 1.063 | 95.0% | 7.122 |
| stochastic | short_gap | No process noise | 3.325 | 2.295 | 27.8% | 2.266 |
| unmodeled_pulse | long_gap | Adaptive mixture | 5.112 | 3.052 | 58.3% | 8.574 |
| unmodeled_pulse | long_gap | Fixed SDE | 5.303 | 3.201 | 53.9% | 7.768 |
| unmodeled_pulse | long_gap | No process noise | 6.834 | 4.827 | 10.6% | 1.813 |
| unmodeled_pulse | short_gap | Adaptive mixture | 5.912 | 3.758 | 45.6% | 7.656 |
| unmodeled_pulse | short_gap | Fixed SDE | 5.946 | 3.786 | 43.7% | 7.146 |
| unmodeled_pulse | short_gap | No process noise | 6.788 | 4.670 | 14.1% | 2.333 |

## Interpretation

- Adaptive scale inference largely suppresses unnecessary process noise on deterministic truth, avoiding the fixed SDE's wide intervals.
- On stochastic truth it improves position error and CRPS relative to both matched controls in these realizations. Some intervals still under-cover; there is no demonstrated calibration guarantee.
- The unmodelled deterministic force produces a nonzero inferred scale and poor coverage. Effective process noise can absorb model mismatch and cannot be interpreted as proof of physical randomness.
- Per-group plots show standard deviation over three seeds, not confidence intervals. Schedules for one seed are correlated; frames, coordinates and posterior samples are not independent replicates.
- A separate nonlinear likelihood audit is needed: small per-pixel linearization error can accumulate into a large full-image likelihood difference. The current evidence is exact only for the local linearized model.

## Artifacts and reproduction

The self-contained report and authoritative per-case JSON are in `examples/scale-study/`. The exact study plan is `configs/scale_study.json`; method details are in `docs/ADAPTIVE_NOISE.md`. The global source fingerprint identifies the historical implementation. Later audit code does not retroactively change these measurements.
