# Verified benchmark campaign results

The campaign improved the original prediction pipeline on both datasets. SoftVT also beats the geometry-matched rigid control on the new test. PhysTwin remains behind stronger previously available controls, so its new branch is an ablation, not a replacement for the best previous benchmark result.

| Evaluation | Original | New | Change | Coverage |
|---|---:|---:|---:|---|
| SoftVT, 40 new Phase B episodes, coordinate RMSE | 18.932 mm | 17.865 mm | 5.64% lower | 40/40 episodes; 1,998/1,998 forecast frames |
| PhysTwin, 8 historically exposed cases, manual-track mean | 43.426 mm | 41.277 mm | 4.95% lower | 8/8 cases; no failed forecast frames |

SoftVT improves 36 cases and worsens 4. Its rigid control scores 18.543 mm, so the new method improves that control by 3.66%. Aligned shape error is essentially unchanged: this is primarily better object motion and release prediction, not demonstrated better elasticity inference.

PhysTwin improves 3 cases and leaves 5 byte-exactly unchanged. Stronger archived controls remain better: the prior exploratory selector scores 33.019 mm, position ridge 29.705 mm, and official PhysTwin 21.144 mm. Those controls were not retrained in this campaign. The large unchanged single-lift-zebra error remains a major weakness.

## What changed

SoftVT estimates slip from the last eight active observed frames in the controller frame, then corrects release motion. A separate Phase B rule uses a prefix-calibrated rigid/contact fallback only when the original trajectory contains nonfinite values. It does not switch using future error. All 179 previously finite regression forecasts remain bitwise identical to Phase A. The only fallback in the 220-case evaluation repairs object-soft-task2-demo31. All 220 proposed trajectories have complete forecast coverage.

The original Phase A result is preserved: 140 historical cases improve 20.978 to 19.735 mm. Phase B also includes the earlier 40 episodes, now exposed regression cases, with 40/40 complete proposed trajectories. Their original baseline has missing frames, so its available-frame mean must not be treated as a complete paired comparison. The fresh Phase B table above has complete coverage for both methods.

PhysTwin fits the width of Gaussian attachment weights jointly with the inherited mechanics parameters. An independently shortened-prefix fit must improve validation by more than 2% before the new full-prefix prediction replaces the original. Missing or invalid comparisons retain the original exactly. Optimizer checkpoints are not all converged optima. The FEM anchor and the empirical output adapter are distinct: final adapted outputs can still invert tetrahedra. These results do not establish physical admissibility, calibrated material estimates, or reliable uncertainty.

## Other tested directions

The separately frozen, always-on Gaussian residual family scores 43.104 mm over all eight PhysTwin cases: 5 improve and 3 worsen, only a 0.74% mean gain over the original. It is not selected per case after scoring, nor combined retrospectively with the width method.

A history-aware direct-support ridge model improved prefix validation and development coordinate/shape metrics but worsened development manual tracking from 34.076 to 65.984 mm; it was rejected. The moving-carrier plus Gaussian deformation model scored 35.341 mm on that development case versus 34.076 mm originally. Its full eight-case freeze/evaluation was not completed before interruption. Its source and proposal are published as unfinished development work, not benchmark evidence. Earlier development diagnostics and rejected branches are retained. This was a multi-candidate development campaign, not a single preselected successful hypothesis.

## PhysTwin case results

| Case | Original manual mean (mm) | Width primary (mm) |
|---|---:|---:|
| single_lift_sloth | 34.076 | 33.716 |
| double_stretch_sloth | 12.123 | 12.123 |
| single_lift_dinosor | 18.119 | 18.119 |
| double_stretch_zebra | 13.115 | 13.115 |
| single_lift_zebra | 130.996 | 130.996 |
| double_lift_sloth | 29.372 | 28.817 |
| single_push_sloth | 70.188 | 53.906 |
| double_lift_zebra | 39.423 | 39.423 |

## Videos

These are synchronized forecast-only point-cloud views with shared world axes and a moving camera, not photorealistic reconstructions. PhysTwin's left panel contains observed reconstructed points, not exact physical ground truth. Cases were fixed before scoring; numerical failures remain visibly marked.

- [SoftVT new episode: actual / original / robust](robust-videos/object-soft-task1-demo40.mp4)
- [SoftVT failure repair: actual / original / robust](robust-videos/object-soft-task2-demo31.mp4)
- [PhysTwin development lift: observed / original / width / Gaussian](phystwin-videos/single_lift_sloth.mp4)
- [PhysTwin stretch: observed / original / width / Gaussian](phystwin-videos/double_stretch_sloth.mp4)

## Verification and scope

Three independent audit implementations passed 8,417 checks: source/input/output hashes, fixed cohort counts, regression preservation, coverage, paired summaries, and independent metric recomputation. PhysTwin raw metrics were recomputed for all eight cases; SoftVT aggregate arithmetic covers all 220 cases and raw metrics were independently checked on prespecified cases. Audit code was prepared by a separate agent; the final post-interruption execution was performed by the main agent. This is independent arithmetic/source verification, not a fresh optimizer replication. All 385 pinned historical files remain unchanged. The final test log is [finish-tests.txt](finish-tests.txt).

Inputs include initial mechanical geometry, observed-prefix 3D positions and prescribed future controller positions. No true material or force labels are supplied. SoftVT's new episodes reuse two assets and do not demonstrate independent-object OOD generalization. All eight PhysTwin cases were historically exposed; their results are exploratory. Descriptive episode bootstrap intervals should not be interpreted as independent-object confidence intervals. Metrics differ by dataset and are not pooled.

The original campaign was interrupted by a Codex usage-limit error. Background fitting, scoring and rendering finished. The user then authorized verification and publication on 1 November 2026. No fitting or new forecasts were launched during this finish step; the original campaign deadline and frozen sources were not changed. Raw datasets and prediction arrays remain local. [Reproduction instructions](REPRODUCE.txt) describe compressed reports and post-campaign audit entry points.
