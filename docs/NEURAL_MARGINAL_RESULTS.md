# Neural SDE and marginalized forces: completed sparse-view comparison

The combination is implemented and all **36 fixed fits** are complete. Structured balanced internal noise improves missing-motion reconstruction on the stochastic sequences, but the current neural amplitude model does **not** show an advantage over a constant-amplitude control. The local uncertainty approximation is inadequate on most added-noise fits, and a separate evaluation audit exposes poor network calibration in the operating region. These limits are findings, not omitted failures.

[Open the offline report](../examples/neural-marginal/report.html), [method and equations](NEURAL_MARGINALIZATION.md), and [calibration audit figure](../examples/neural-calibration-audit/calibration.png). Exact inputs, reference motion/audits, calibrated weights, all solver states, force distributions, per-seed results and CSV accompany the report. Historical v0.6 evidence remains unchanged.

## What was compared

Every fit uses the same corrected six-substep staggered integrator, continuous 128-slab renderer, fixed external-force prior, known scene, image noise, and prior-mean zero-load initialization. Six 24x24 RGB images from one moving known camera are observed at frames 0,3,6,14,17,20; frames 7-13 form the missing gap. All observations on both sides are used for offline smoothing, rather than online forecasting.

- **Current floor:** the existing full-rank independent free-node discrepancy and marginalized force.
- **Constant internal:** add balanced central edge noise with a rank-three spatial kernel and constant amplitude calibrated independently.
- **Neural internal:** add the same internal-noise structure with amplitude from the frozen 2 -> 8 -> 1 MLP.

The constant control still follows evolving edge directions, so its covariance depends on geometry. It isolates amplitude adaptation rather than replacing physics with fixed world-space random forces. All retain the same declared full-rank floor. The neural weights are trained on separate synthetic innovations; they are not inferred from test video, and their uncertainty is not propagated.

Four conditions and three seeds give 12 condition/seed datasets and 36 fits. Conditions share matched underlying seeds/noise, so pooled stochastic results are six paired condition/seed cases, not six independent realizations. The three seeds are exploratory; there is only one geometry and one sparse schedule.

## Missing-motion reconstruction

Numbers are mean per-seed coordinate RMSE over free vertices and missing-gap timestamps, in millimetres. Individual seeds are plotted and recorded, including those where added noise worsens the estimate.

| Truth | Current floor | Constant internal | Neural internal |
| --- | --- | --- | --- |
| no_load | 2.815 | 2.456 | 2.458 |
| external_pulse | 2.824 | 2.762 | 2.789 |
| stochastic | 4.778 | 2.721 | 2.788 |
| pulse_and_stochastic | 4.916 | 3.066 | 3.160 |

Pooling the two stochastic conditions, mean gap error is **4.847 mm** for the current model, **2.893 mm** with constant internal noise, and **2.974 mm** with neural internal noise. Those are descriptive reductions of about 40% and 39%, respectively. The constant control is slightly better on these averages; three seeds cannot establish a universal ranking. The small neural/constant differences also need posterior/time-refinement validation before being interpreted as physical improvements.

On deterministic seed 4101, added noise worsens reconstruction: no-load error rises from 1.012 to 1.358 mm with neural noise; pulse error rises from 1.137 to 1.905 mm. Improvements on other seeds bring the deterministic averages closer. Extra flexibility can explain away evidence about forces/material instead of always helping.

## Uncertainty and external-force estimates

All **36/36** fits pass the predeclared optimizer stationarity screen. That is separate from posterior validity. The independent 128-proposal nonlinear weight-concentration screen passes **12/12** current-floor fits, **2/12** constant-internal fits, and **2/12** neural-internal fits. Even a pass is not proof of adequate posterior exploration or calibration.

| Truth | Model | Gap CRPS mm | Local 90% coverage | Local width mm | Force RMSE N | Weight screens passed |
| --- | --- | --- | --- | --- | --- | --- |
| no_load | floor | 1.678 | 87.3% | 9.56 | 0.01407 | 3/3 |
| no_load | constant | 1.491 | 99.6% | 12.57 | 0.01350 | 0/3 |
| no_load | neural | 1.490 | 99.6% | 12.40 | 0.01352 | 0/3 |
| external_pulse | floor | 1.676 | 87.3% | 9.55 | 0.01515 | 3/3 |
| external_pulse | constant | 1.639 | 98.8% | 12.57 | 0.01428 | 1/3 |
| external_pulse | neural | 1.647 | 98.0% | 12.39 | 0.01433 | 1/3 |
| stochastic | floor | 2.883 | 70.2% | 9.51 | 0.01847 | 3/3 |
| stochastic | constant | 1.590 | 99.6% | 12.55 | 0.01369 | 0/3 |
| stochastic | neural | 1.619 | 99.6% | 12.37 | 0.01380 | 0/3 |
| pulse_and_stochastic | floor | 2.976 | 65.5% | 9.50 | 0.02037 | 3/3 |
| pulse_and_stochastic | constant | 1.778 | 98.4% | 12.55 | 0.01497 | 1/3 |
| pulse_and_stochastic | neural | 1.822 | 98.0% | 12.37 | 0.01513 | 1/3 |

Adding internal noise improves the empirical local Gaussian CRPS on stochastic truth, but the intervals are wider and often over-cover. Coverage is over correlated components from only three seeds. These local predictive scores do not override failed nonlinear proposal diagnostics and do not establish calibrated force or motion uncertainty.

The primary force score uses the local trajectory-mixture force mean. Because that trajectory approximation is flagged, a SECONDARY post-fit diagnostic also holds motion/material at their marginal MAP and evaluates the exact conditional force mean. It is not a replacement primary score or a full posterior mean. All conditions/models/seeds are retained. On stochastic-only truth its mean force RMSE is 0.01852 N (floor), 0.01271 N (constant), and 0.01285 N (neural). On push-plus-stochastic truth these are 0.02042, 0.01412 and 0.01431 N. This supports improved point force attribution from internal-noise structure, with no clear neural-specific gain.

The total force covariance correctly includes E[S_a(z)] + Cov[m_a(z)]; unlike v0.6, S_a varies with state and is averaged. Gaussian force removal remains exact conditional on the trajectory. The trajectory posterior and frozen network model remain approximations.

## Why the network behaves like the constant control

Post-fit MAP diagnostics show neural amplitude coefficient of variation only **1.18-1.24%** over solver edge states. Its mean is about 0.002458, compared with the separately calibrated constant 0.002567. Thus the current neural function barely modulates amplitude in this operating region. The control matches RMS amplitude on independent broad calibration features, not on test states; no test-based variance matching or tuning was performed.

The secondary evaluation-only audit queries the known analytic noise law on TRUE saved frame states, using both positions and physical velocities. The network's relative amplitude RMSE is **36.3%**, versus **14.4%** on its separate broad calibration features. The constant control is **42.1%** on these true states. All evaluated features lie within the original training ranges. True mean amplitudes are around 0.00182, while neural predictions are around 0.00246; true amplitude variation is about 5.3-5.7%, while neural variation is about 1.2%.

This is consistent with inadequate calibration/underfitting in the operating region, even without out-of-range inputs. It helps explain weak adaptation and conservative noise levels. It does not prove that a better calibrated neural SDE could never help. No weights, hyperparameters, inference results or primary metrics were changed after this audit. The reproducible script is `scripts/audit_neural_calibration.py`; raw audit metrics and reference hashes are in `examples/neural-calibration-audit/summary.json`.

## What force marginalization contributes

This 36-fit study holds force marginalization FIXED and varies the noise model. Its gains therefore should not be attributed to marginalization itself. The earlier v0.6 experiment separately compared alternating, joint and marginalized solvers.

For an inferred path z, r(z)=G a+eta, with Gaussian a and covariance Q(z) independent of a once the path is fixed. Integrating the force gives C(z)=Q(z)+G G^T and exact force conditional covariance A(z)^(-1), where A(z)=I+G^T Q(z)^(-1)G.

Benefits are removing a separate force optimization block, recovering coherent force distributions/correlations, and accounting for all plausible forces instead of selecting only the best force for each candidate motion. Computation exploits block-diagonal Q and a 15x15 force system rather than explicitly inverting the full residual covariance. The remaining path/material optimization is still nonlinear and currently uses dense state metrics.

The exact relation is:

    marginal_NLL(z) = min_a joint_NLL(z,a) + 0.5 logdet(A(z)) - (15/2) log(2 pi).

With fixed Q/G, A is constant: joint/profile and marginalized trajectory optima coincide, so no automatic accuracy improvement follows. With state-dependent internal covariance, the force-volume correction varies. Profiling a best force and omitting it gives a different probability model. Marginalization neither invents missing-view information nor guarantees faster inference, force identifiability or calibrated uncertainty.

## Reference accuracy and validation

Deterministic truth uses the published RK4-96 trajectories. Stochastic truth uses an independently implemented symplectic Euler-Maruyama integrator with the analytic diffusion law, rather than the fitted network. Brownian-coupled grids are 192/384/768 substeps per frame. Final pairwise coordinate changes are **0.0119-0.0124 mm** across all six stochastic sequences. Net internal impulse, impulse torque, supports, free-node momentum and positive deformation determinants pass the declared gates.

A separate evaluation-only oracle-amplitude/analytic-load audit checks the production staggered solver against the same fine paths. Six-substep coordinate errors are **0.111-0.169 mm**; twelve-substep errors are **0.067-0.085 mm**. The oracle never enters initialization or fitting. These are numerical consistency checks using the same FEM constitutive law, not rigorous absolute SDE error bounds, experimental material validation or mesh convergence.

All **77 checks** and original/new end-to-end smoke commands passed [implementation CI](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39037380351). Tests include dense versus structured state-dependent Gaussian densities/gradients, causal Jacobians, covariance-information derivatives, force total covariance, internal balance, baseline equivalence and independent stochastic refinement.

Implementation: `08c3281825a7730a0ce3e44b1c16f3fe19e218ff`.
Numerical fingerprint: `689baa8640bcd18883d087d1a65f57feb11afd32e446c0250ac926fb71b25695`.
Plan: `configs/neural_marginal_study.json`.
Raw local run: `runs/neural-marginal-final`.

The publisher verifies all 36 saved objectives, point paths/position scores, interval standard deviations, RGB/noise and initial-state identities, calibration/source hashes, reference gates, force covariance identities and portable queries. `publication.json` records artifact hashes. Concurrent reference preparation used identical committed functions and seeds and is recorded in `execution.json`.

Large local precision factors stay outside Git. `scripts/recover_neural_posterior.py` rebuilds a covariance from one published solver state without optimization; the recovered neural precision factor matched its original exactly (maximum difference zero). Geometry, actual spatial basis/sign convention, mesh/masses, kernel times and network weights are included for handover.

## Next scientific work

Use the constant-internal model as a serious baseline. Before claiming a neural-specific improvement, predeclare an independent calibration task concentrated on the relevant physical regimes, test symmetry/energy-based features and amplitude accuracy, and validate nonlinear posterior exploration. Do not retrain on these evaluation states or choose a new amplitude to force this comparison to win. Repeated independent scenes, broader excitation, noise-floor sensitivity, multiple sparse schedules and posterior time refinement are needed before real-video or calibration claims. Unknown geometry/camera, arbitrary force locations and contact remain unimplemented.
