# Structured-force posterior: controlled results

Subsequent [simulation/view audit](SIMULATION_VIEW_AUDIT.md): the shared
historical time step has about 3.3 mm error against independent converged RK4.
Measurements below remain valid relative to their discretized synthetic truth,
but do not establish millimetre-level continuous-time physical accuracy.

Implementation commit `1644958`, version 0.4; a subsequent test-only commit fixes a derivative check at a nondifferentiable depth-order boundary. The executed plan is `configs/force_study.json`, with its exact source fingerprint and numerical environment saved in the report/manifest. Independent condition partitions use the unchanged plan; source/environment/configuration and calibrated network tensor equality are verified before merging at completed-model boundaries. `execution.json` records this orchestration. This is a known-geometry, known-camera synthetic investigation using realization seed 4101. Four force conditions and three conditional models produce 12 fits. Conditions share a realization seed and are not independent calibration replicates.

Each fit uses four chains, discards 500 warmup transitions and retains 1,000 transitions per chain. Every retained state enters latent diagnostics and smooth external-force scores/moments/intervals. An equally spaced 128-state subset is used for image/trajectory scoring and combined finite-window force/impulse evaluation. Stochastic forcing has fixed calibrated unit amplitude; no model probabilities are inferred.

The `no_load` condition retains known gravity, damping and initial motion; it excludes only the extra unknown load and stochastic impulses. Partitioned runs share CPU resources, so their recorded wall times are not a runtime comparison between inference methods.

## Force recovery and predictive scores

All values are finite-chain estimates. Read the convergence screen before using an interval. The SDE-only model excludes the external force, so its zero-width external band is a conditional model assumption, not evidence of force absence.

| Truth | Model | Force RMSE N | Force CRPS N | 90% coverage | Width N | Impulse RMSE Ns | Withheld position RMSE mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_load | force_only | 0.00771 | 0.00494 | 96.7% | 0.04475 | 0.00429 | 0.989 |
| no_load | sde_only | 0.00000 | 0.00000 | 100.0% | 0.00000 | 0.00000 | 1.747 |
| no_load | force_and_sde | 0.00889 | 0.00657 | 100.0% | 0.07450 | 0.00556 | 1.980 |
| external_pulse | force_only | 0.01105 | 0.00692 | 82.2% | 0.03910 | 0.00684 | 1.626 |
| external_pulse | sde_only | 0.01615 | 0.00688 | 33.3% | 0.00000 | 0.00860 | 6.147 |
| external_pulse | force_and_sde | 0.01639 | 0.00908 | 89.4% | 0.04989 | 0.01304 | 3.780 |
| stochastic | force_only | 0.01522 | 0.00891 | 66.7% | 0.03282 | 0.00920 | 3.620 |
| stochastic | sde_only | 0.00000 | 0.00000 | 100.0% | 0.00000 | 0.00000 | 2.162 |
| stochastic | force_and_sde | 0.00997 | 0.00707 | 98.9% | 0.05391 | 0.00247 | 2.068 |
| pulse_and_stochastic | force_only | 0.02155 | 0.01337 | 50.6% | 0.03699 | 0.00663 | 3.535 |
| pulse_and_stochastic | sde_only | 0.01615 | 0.00688 | 33.3% | 0.00000 | 0.00860 | 5.726 |
| pulse_and_stochastic | force_and_sde | 0.01257 | 0.00781 | 100.0% | 0.07286 | 0.00263 | 2.421 |

## Convergence and force-functional checks

The screen requires rank-normalized/folded split Rhat <1.01, bulk ESS >=400, tail ESS >=400 and no frozen chains. The impulse-functional screen is separate. Both refer to numerical exploration, not physical calibration or exclusion of missed modes.

| Truth | Model | All latent checks pass | Flagged quantities | Max defined Rhat | Min defined bulk ESS | External-impulse checks pass |
| --- | --- | --- | --- | --- | --- | --- |
| no_load | force_only | False | 4/24 | 1.051 | 119.1 | False |
| no_load | sde_only | False | 164/189 | 1.067 | 45.6 | excluded component |
| no_load | force_and_sde | False | 204/204 | 1.127 | 29.3 | False |
| external_pulse | force_only | False | 24/24 | 2.367 | 5.0 | False |
| external_pulse | sde_only | False | 189/189 | 3.764 | 4.3 | excluded component |
| external_pulse | force_and_sde | False | 204/204 | 3.843 | 4.3 | False |
| stochastic | force_only | False | 24/24 | 3.336 | 4.4 | False |
| stochastic | sde_only | False | 189/189 | 2.994 | 4.5 | excluded component |
| stochastic | force_and_sde | False | 204/204 | 3.843 | 4.3 | False |
| pulse_and_stochastic | force_only | False | 24/24 | 1.079 | 83.3 | False |
| pulse_and_stochastic | sde_only | False | 68/189 | 1.051 | 78.0 | excluded component |
| pulse_and_stochastic | force_and_sde | False | 204/204 | 1.142 | 22.9 | False |

None of the twelve fits passes the complete screen, and no inferred-load model passes all three impulse-functional checks. The force-only pulse fit has lower point/trajectory error than the excluded-force baseline, but its Rhat and ESS rule out treating those finite-chain intervals as reliable. This experiment does not establish a superior force estimator or calibrated coverage.

On purely stochastic truth, a smooth-load-only model produces a nonzero force estimate even though the true extra external load is zero. Missing stochastic dynamics, material/initial-state ambiguity and incomplete sampling can all affect that estimate. The combined model explicitly represents both mechanisms, but still fails the exploration screen. A 100% coverage figure accompanied by broad bands is not evidence of calibration.

## Representation and physical validity

- no_load: temporal-basis oracle force RMSE 0.000000 N. This diagnostic uses truth only after inference; it never chooses the basis or hyperparameters.
- external_pulse: temporal-basis oracle force RMSE 0.002870 N. This diagnostic uses truth only after inference; it never chooses the basis or hyperparameters.
- stochastic: temporal-basis oracle force RMSE 0.000000 N. This diagnostic uses truth only after inference; it never chooses the basis or hyperparameters.
- pulse_and_stochastic: temporal-basis oracle force RMSE 0.002870 N. This diagnostic uses truth only after inference; it never chooses the basis or hyperparameters.
- Retained-phase invalid proposals: 0, counted as rejections rather than replaced draws. Minimum deformation determinant over rendered samples: 0.876885.

The temporal-basis oracle error is below the measured pulse force error, so basis restriction alone does not explain the result. The next numerical step should separate proposal failure from model ambiguity: compare a gradient-free sampler, examine renderer depth-order changes, and test stable automatic-differentiation linearizations or a smoother rendering likelihood. Revalidate the posterior before expanding the spatial force basis or inferring camera/geometry. More sampling by itself is not assumed to solve these problems.

## Reusable posterior output

Per-model `fields.json` files are included in `examples/force-study/cases/`. They contain inducing times, spatial weights, force-prior settings, coefficient moments from every retained state and coherent coefficient draws. The published snapshots additionally attach an `inference` status/provenance block derived from the matching case record, with a hash of the original raw field file. Numerical force distributions are unchanged. Use `physgauss.forces.evaluate_force_posterior(fields, times)` to obtain total/nodal force means, joint cross-time covariance and function draws within the recording interval. Read `fields['inference']` before using a published snapshot: portable fields do not turn a flagged posterior into a reliable one.

The profile is fixed and mass-proportional on this cube; there is no learned force location or arbitrary field topology. Known mass/density and metric camera scale provide N units. Material, force and initial-state ambiguity remains. The neural diffusion prior was calibrated on independent synthetic innovations, not learned from these images. More realizations, geometries and calibration tests are needed before a real-world claim.

See [the exact method and query API](FORCE_POSTERIOR.md) and [the self-contained force report](../examples/force-study/report.html). Raw paired force/impulse samples, observation tensors, references and chains remain in ignored `runs/force-study-v2`; compact distributions, covariance summaries and measurements are committed.

## Reproduction

```sh
physgauss check
physgauss forces --plan configs/force_study.json --output runs/new-force-study
physgauss forces --plan configs/force_study.json --output runs/new-force-study --resume
```

The executed local path is `runs/force-study-v2`. Use a new output directory in an existing checkout; an older interrupted development run may occupy `runs/force-study`. Source/environment/plan guards prevent mixing incompatible results. Resume restarts an unfinished model from its seed and skips completed model records.
