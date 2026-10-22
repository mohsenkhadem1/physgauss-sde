# Current priority: measurable deformation and resolved mechanics before learning

The sealed 32-object prediction study remains complete and unchanged. The current work asks whether the data contain useful elastic deformation, whether the nonlinear forward problem is numerically resolved, and whether a small conditional posterior can be checked independently. **No nonlinear inverse fit, neural training or active-probing expansion is authorized by these results.**

## Population motion is predominantly rigid

A frozen retrospective audit includes **all 32 fresh objects, both A/B programs and all 61 saved frames**. Each frame is compared with its best proper rotation and translation using both uniform-node weights and reference-volume lumped mass weights. Homogeneous density cancels from the normalized mass weights; no material or support labels were needed.

For the full paired A/B trajectories, averaging complete-object coordinate RMS:

| Weighting | Total motion | Nonrigid residual | Mean objectwise residual/total RMS ratio |
|---|---:|---:|---:|
| Uniform nodes | 1.2043 mm | 27.74 µm | 3.36% |
| Reference-volume lumped mass | **1.1353 mm** | **24.75 µm** | **3.19%** |

Coordinate RMS divides weighted squared vector displacement by three coordinates and the number of saved frames. Each ratio uses that object's own total motion as denominator; averaging these ratios differs from dividing the two population means. The mass-weighted mean nonrigid RMS has a 95% object-bootstrap interval of **18.17–31.81 µm**. On the common A31–60/B0–60 windows it is 25.87 µm, with a mean objectwise ratio of 2.96%. Objects, with their paired programs, are the bootstrap units; frames, vertices and tetrahedra are not independent samples.

Compatible tetrahedral deformation gradients give mean volume-weighted principal-stretch RMS deviation **0.0430%**, with a largest local absolute deviation of **3.66%**. Local stretches span 0.96343–1.03430 and Jacobians 0.97603–1.02298. Small whole-field deformation therefore does not mean every element is unstrained. These are population findings, not an extrapolation from an earlier single-case diagnostic.

Evidence: [frozen protocol](../../operator-dev/point-rotation-population-deformation-v01/protocol.json), [population summary](../point-rotation-population-deformation-v01/summary.json). Six synthetic tests cover rigid motion, affine strain, weighting, proper rotations and aggregation. The audit completed in 17.42 s on two CPU threads; per-object transforms/statistics remain local.

## The acquired depth pixels also have weak elastic contrast

A separate frozen diagnostic renders the **actual fixed A0.5/A1.0 s queries** for each true FEM surface and its best volume-weighted rigid-fit surface. These are noiseless camera-axis depths; only paired-visible differences are compared with the stipulated single-observation depth-noise standard deviation of 200 µm.

| Acquired window | Mean objectwise paired-depth RMS contrast | Paired foreground / total pixels |
|---|---:|---:|
| A0.5 s | 7.45 µm | 448 / 1,152 |
| A1.0 s | 46.30 µm | 416 / 1,152 |
| Both frames | **32.64 µm** | **864 / 2,304** |

The combined mean has a 95% object-bootstrap interval of **25.63–40.50 µm**; pooled paired-depth RMS is 39.07 µm. All remaining **1,440 pixels are background in both surfaces**, and there are **zero visibility changes**. Four of 864 paired foreground rays exceed 200 µm contrast; the largest is 283.09 µm. Every object's paired-ray RMS is below that noise standard deviation.

This supports weak elastic contrast at the acquired pixels, **not zero information**. Multiple observations, dynamics, priors and rigid-body response can still carry information. The contrast is not an estimate of total information or material identifiability. Missed rays are never replaced with artificial measurements, and XYZ residual RMS was not equated with range-space signal-to-noise.

Evidence: [separate protocol](../../operator-dev/point-rotation-base-depth-signal-v01/protocol.json), [depth summary](../point-rotation-base-depth-signal-v01/summary.json). Six synthetic tests passed; all 32 objects were retained. CPU execution took 0.94 s, without GPU use, inference or noise draws.

## Stronger nonlinear excitation exposes an unresolved forward problem

A separate development benchmark uses balanced Gaussian patch forces to generate finite strain. It is not a rerun or alteration of the sealed population. Initial temporal and spatial gates failed, and a coarse useful-strain result did not transfer to refined meshes.

A new sparse implementation of the **same** finite-strain mechanics agrees with the frozen Torch trajectory at 128 tetrahedra to **4.16×10⁻¹⁷ m** maximum coordinate error. This is useful implementation parity, not physical resolution.

| Further mesh comparison | Held-node trajectory RMS difference | Declared tolerance | Result |
|---|---:|---:|---|
| 128 → 1,024 tetrahedra | 0.38117 mm | 0.23133 mm | FAIL |
| **1,024 → 8,192 tetrahedra** | **0.38034 mm** | **0.24388 mm** | **FAIL** |

The error does not meaningfully contract. The finest trajectory has **11.52% peak absolute principal-stretch strain and 0.800 mm peak nonrigid RMS**, missing the original 2–8% strain/at-least-1 mm excitation target. Positive Jacobians, residual and work checks pass, but no mesh qualifies. Conditional temporal continuations were therefore not launched. No preferred mesh, relaxed tolerance, nonlinear fit or network result substitutes for the failure.

The [initial development report](../../operator-dev/nonlinear-excitation-v01/REPORT.md) and [sparse-resolution report](../../operator-dev/nonlinear-resolution-v02/REPORT.md) retain every failure. A local 2/16/128-tet clip shows the same control/material/support at true scale, with 10× slower playback and a persistent **MESH COMPARISON—UNCONVERGED** label. It is not an actual-versus-learned video or a continuum reference. [Compact render metadata](../nonlinear-mesh-video-v01/render-manifest.json) are publishable; MP4/PNG remain local.

## A small conditional numerical posterior is now independently checked

The manufactured compliant-fixture diagnostic has a positive **two-dimensional conditional** result: E and finite fixture stiffness K are unknown, while fixture center **0.029 m** and width **0.031 m** are fixed and known. It uses the same finite linear model, 12 manufactured displacement observations, 20 µm noise and original intrinsic prior. It is **not the original four-dimensional marginal posterior**.

Independent adaptive integrations agree, and eight prior-started continuous MCMC chains pass: maximum rank/folded split R-hat **1.00220**, minimum bulk/tail ESS **2,763/3,095**. Their CDF discrepancy against the adaptive reference is 0.01684.

A separately frozen higher-particle SMC resolution study also passes: **all six runs** at 8,192/32,768 particles × three seeds meet the original reference/MCMC criteria, and **all 15 cross-run comparisons pass**. Maximum SMC CDF discrepancy from adaptive integration is 0.01426, normalized-coordinate quantile discrepancy 0.004715, and worst future-A/reset-B mean differences are 1.334/1.061 µm. Four completed runs were reused; only the two previously unfinished seeds were continued. No physical model or likelihood changed.

The old 625-point and 83,521-point four-dimensional grid failures, lower-particle SMC instability, v01 NumPy-boolean reporting failure and v03 180-second cap remain preserved. Passing this conditional linear manufactured problem does not qualify nonlinear dynamics, unknown fixture geometry, physical calibration or neural inference. See the [conditional reference method and limits](../../../docs/GAUSSIAN_COMPLIANT_CONTINUOUS_REFERENCE.md) and [final higher-particle summary](../compliant-smc-resolution-v04/summary.json).

## What comes next

1. Check physical central-force alignment: prescribed Gaussian patch centers versus realized quadrature force centroids and measured moments, without silently changing spatial loads across meshes.
2. Resolve the nonlinear mechanics and useful-strain operating regime under frozen spatial/temporal criteria.
3. Only then run a small nonlinear posterior with an independent numerical reference, before boundary-learning networks or intelligent probing.

No calibrated boundary-learning claim follows yet. The [sealed rotation-aware prediction study](../point-rotation-analysis-v02/REPORT.md) remains the latest prediction study; it is neither replaced nor retrospectively retuned by these diagnostics. Raw trajectories, observations, particles, chain draws and videos remain local.
