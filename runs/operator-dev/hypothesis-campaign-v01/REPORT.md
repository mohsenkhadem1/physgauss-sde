# Fixed excitation and supplied-physics engineering checks

Two globally fixed 120 Hz world-force programs independently excite both handles in xyz. Each is bounded by 0.02 N per handle, active 0.15–2.55 s, with 61 outputs over 3 s. Their six-dimensional physical force Grams have condition numbers 61.57 and 19.54 (declared limit 100). Integer private refinement repeats exact float64 commands; no latent-conditioned amplitudes or waveform resampling occur.

The finite TRAIN design crosses 4 complete public hardware contexts, 4 BCs, 3 shared joint material tuples, and 2 programs: 48 hypotheses and 96 capture slots. E uses equal-log-width strata with conditional density factor 3; rho/nu/drag keep their original marginals. Sixteen support/motion/nuisance states are reused across materials. This is a dependent counterfactual cohort, not iid prior data or identification of four separate material effects. Only 2 unique slots have engineering captures so far; point-pin refinement creates 4 total trajectories including moving 120 Hz.

A private 1.25 optimizer metric converged the preserved difficult one-step diagnostic in 39 iterations at 7.63e-11 N without changing its physical objective. Prior failures remain preserved. All 4 new finite trajectories independently pass mechanics checks. For H0/m0/point-pin/programA, full 61-frame coordinate refinement errors are 13.851→8.191µm (ratio 0.5913), below 44.258 µm. Common 120 Hz is an economical engineering choice; this does not temporally qualify all 96 slots. Moving120 has no temporal refinement.

| Frozen linear-physics baseline | Position RMSE | Relative to finite motion | Construction + prediction |
|---|---:|---:|---:|
| pointpin120 | 13.093µm | 1.483% | 4.413s |
| pointpin240 | 13.191µm | 1.492% | 8.117s |
| pointpin480 | 13.255µm | 1.497% | 14.889s |
| moving 120 Hz | 22.653µm | 0.817% | 5.104s |

Every baseline uses the same supplied material/full hardware/support law and exact known commands, with no fitting or response input. The finite runtime logs are 95.23/244.72/485.99/133.82 s respectively; these are not controlled speed benchmarks. Linear rotation causes 38–42 µm rigid-patch distance defects; neither model inverts in these comparisons. Small errors make this physical baseline mandatory; these cases alone do not justify a learned residual. C21 neural limitations remain preserved.

The new dataset builder requires the complete declared 96-slot index and excludes IDs/private rates from model features. No crossed corpus or new predictor is admitted by this checkpoint. Raw predictions, finite trajectories and command binaries stay LOCAL. Sparse-view inference, calibrated uncertainty, probing and native/continuum equivalence remain unresolved.
