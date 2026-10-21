# Numerical reference audit for the recovery campaign

Known-force material calibration remains blocked by numerical sensitivity of the
current Isaac reference. These are TRAIN diagnostics, not a new predictive
comparison. No reserved control gate, calibration response, or fresh
confirmation response was used predictively. No existing simulator, forward-model or
training source was modified by these experiments.

## What the completed experiments establish

The measured forces combine as intended, stationary zero-force responses are
about 4 nm RMS, and fresh 960 Hz trajectories reproduce the earlier trajectories
bit for bit. This establishes repeatability at one configuration, not numerical
convergence or a physically calibrated constitutive law.

For TRAIN t0001, the unchanged material, supports, mesh, hardware, 31 sample
times and diagonal pulse give:

| Physics rate | Diagonal coordinate RMS from rest | Minimum saved volume ratio |
|---|---:|---:|
| 480 Hz | 111.255 µm | 0.98737 |
| 960 Hz | 236.556 µm | 0.94060 |
| 1920 Hz | 148.370 µm | 0.96569 |

The 480→960 Hz trajectory difference is 181.216 µm coordinate RMSE; the
960→1920 Hz difference increases to 275.427 µm. Maximum nodal differences are
2.917 and 3.937 mm. Force-impulse quadrature agrees with the analytic pulse
within 1.38e-11 N·s. Recorded clocks exactly equal the number of steps times the
representable float32 timestep. The World timestep getter separately returns
the nominal double value; this distinction is recorded rather than concealed.
All original sensor, geometry, support and force checks pass.

The initial base validator reached its final reset check and raised
`KeyError('poke_0')`: the numerical menu names that candidate `zero_force`.
Those failures remain in the execution ledger. A separate, fully saved validator
adapts only its two reset-reference dictionary keys. It retains every original
clock, force, source, geometry, camera, ray, capture-hash and replay check and
every tolerance. No images or physics were regenerated for this adaptation.

## Iteration and coordinate factorial

Six cells were declared before observing their responses: 960 Hz with 32, 64,
128 deformable iterations, in the original coordinates and with every physical,
reference, support, handle and camera coordinate translated by −0.25 m in z.
Local attachment offsets, inertia and camera rays are invariant under this
translation. Rigid solver iterations remain at their composed fallback of 16.

| Condition | Iterations | Diagonal RMS | Status |
|---|---:|---:|---|
| Original | 32 | 236.556 µm | Full independent QC passed |
| Original | 64 | 682.998 µm | Full independent QC passed |
| Original | 128 | — | Compression guard failed at 1.436458 s |
| Recentered | 32 | 39.694 µm | Full independent QC passed |
| Recentered | 64 | 152.797 µm | Full independent QC passed |
| Recentered | 128 | 886.273 µm | Full independent QC passed |

The original 128-iteration failure has minimum tetrahedral volume ratio
0.097525, maximum nodal displacement 27.27 mm and fixture error only 6.7 nm.
It is preserved, never replaced or omitted as an unsuccessful scene. In the
recentered condition, successive iteration differences increase from 136.776
to 838.901 µm RMSE. This is not a converged operating point.

At matched iteration counts, undoing the world translation leaves 233.214 µm
(32 iterations) and 662.680 µm (64 iterations) differences. In this initial
factorial original and recentered conditions ran on GPUs 0 and 3, respectively.
A separately frozen crossed-device 32-iteration replay reproduces each
coordinate condition bitwise after swapping GPUs 0 and 3, including full
trajectory archive hashes for zero, diagonal and reset responses. Assignment
to those two devices therefore does not explain the coordinate difference in
this case. See `numerical-solver-crossed-v01/cross-device-comparison.json`; the
six-cell source and results remain immutable.

Every successful cell has nine initial warmup renders and 93 synchronous
captures with bitwise unchanged observed nodal q/v, rigid poses and velocities,
clocks, step counts and timestep settings around the rendering operation. This
does not expose every hidden warm-start cache, force cache or kinematic flag.
The composed deformable sleep/settling thresholds and settling damping are
authored zero, as are rigid sleep/stabilization thresholds. Scene TGS and
disabled stabilization are recorded. The public rigid sleep getter reads USD;
it is not independent native-actor verification.

A separate physics-only experiment removes the combined intermediate
display-update, zero-time rendering, annotation readback and image-writing
block, retaining initialization and the same physical-state getter timing.
Its q, v, handle positions, commands and complete trajectory archives are
bitwise identical to the captured original 32-iteration case for zero force,
the diagonal and reset replay. It produces **no images and no sensor-QC
certificates**. This intervention did not change that trajectory; startup
effects and other hidden mechanisms remain untested.

## Nonlinear approximation and missing hardware context

Eleven matched pulse controls (three axes plus eight off-axis controls at two
magnitudes) identify a full-rank nine-column quadratic force design with
condition number 19.68. A measured zero-force trajectory supplies its fixed
offset. The independent TRAIN diagonal is opened only after the linear and
quadratic coefficients and predictions have been frozen; it selects nothing.

| TRAIN layout | Force-induced diagonal RMS | Linear error | Quadratic error |
|---|---:|---:|---:|
| t0000, point pin | 23.242 µm | 17.055 µm | 10.304 µm |
| t0001, fixed patch | 236.556 µm | 178.608 µm | 123.677 µm |
| t0002, two patches | 53.625 µm | 40.369 µm | 24.084 µm |
| t0003, moving patch | 320.340 µm | 207.756 µm | 230.518 µm |

Added polynomial capacity helps three layouts but worsens the fourth and does
not resolve the reference problem. Maximum local principal-stretch deviations
are 1.9%, 10.0%, 3.0% and 21.5%, so small whole-body motion does not guarantee
small local strain. PhysX documents a rotationally invariant corotated
elasticity model; finite-deformation effects are plausible. These observations
cannot separate that mechanism from the demonstrated numerical sensitivity.
See [PhysX deformable volumes](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/DeformableVolume.html)
and the [deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html).

A geometry-only audit of 44 fitting TRAIN layouts finds 12 same-handle-index,
same-center groups with different nine-node attachment sets (26 handle
instances), with authored inertia ratios up to 4×. No naturally repeated pair
of both centers was found in that subset; these scenes do not demonstrate
natural duplicates of the entire physical vector.

A controlled t0000 pair changes only the second handle's patch to a valid,
disjoint nine-node patch observed in t0026. Both centers, full encoded 25D
vector, recipe, supports, material, force and camera remain identical. Both
variants pass full QC and have identical zero-force responses. Their diagonal
responses differ by 6.972 µm RMS, about 30% of the original 23.242 µm signal,
with 187.669 µm maximum nodal difference. Their motion amplitudes are nearly
identical, so amplitude alone misses this effect. This establishes a
response-relevant omitted hardware variable in one pair, not a cause of all
forward error or of within-layout nonlinearity.

A future common known-hardware context must include both handles' reference
anchor coordinates, attachment node IDs, corresponding reference local
offsets in the **same node order**, mass, authored inertia and initial
orientation. Use actual common reference geometry; if sorting IDs, permute
offsets together. Supply the same information to every baseline and policy.
Alternatively these quantities must remain explicit latent nuisance variables
beyond the existing 25D representation. A centers-only designed prior must not
be described as the exact Isaac conditional prior.

## Evidence, sources and tests

- `runs/gaussian-campaign-v01/numerical-timestep-v01/`: three rates, immutable
  original errors, source-derived validator, input lineage and analysis.
- `numerical-solver-coordinates-v02/`: six-cell protocols, composed attributes,
  warmup/capture guards, five validations and preserved compression failure;
  `failure-aware-analysis.json` includes the failed cell explicitly.
- `numerical-solver-crossed-v01/`: separately declared crossed-device replay.
- `numerical-capture-backaction-v01/`: physics-only records, exact source diff,
  no-images/no-sensor-QC qualification and bitwise comparisons.
- `numerical-attachment-audit-v02.json`, `numerical-hardware-pair-v01/`,
  `numerical-force-polynomial-four-v01/`, `numerical-rich-axis-replay.json`:
  hardware ambiguity, paired effect, held-diagonal diagnostics and replay hashes.
- `reports/gaussian-campaign-numerical-v03/`: static scientific PNG/PDF and
  manifest; generated by `experiments/campaign_numerical_report.py`.

All new implementation lives in `experiments/campaign_numerical_*`; frozen
workers/configurations have additional provenance beside the unchanged base
source. Tests cover identifiable cross-force terms, measured moving-support
offsets, rigid-rotation strain invariance, reserved-role rejection, original
preservation, midpoint impulses and float32 clocks, reset-guard retention,
attachment overlap prevention, physical/camera translation equivariance,
clock-preserving state mutation rejection, and explicit removal of sensor
claims from the physics-only derivative:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_campaign_numerical_*.py' -v
```

The next physical-calibration claim requires the separately predeclared
numerical gate across all four support types. Passing a single numerical
factorial, even if achieved later, would not establish mesh convergence or
real-world material accuracy.
