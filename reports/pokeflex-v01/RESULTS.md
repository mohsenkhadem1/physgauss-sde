# PokeFlex v01: results

Held-out pokes on 8 PokeFlex objects. The protocol is in `scripts/pokeflex_v01/protocol.json`, the forward model in
`realfem-dev.md`, and the stage-B adapters in `stageB-adapters.md`. The tables below come from `scores.json` (v01),
`scores-stageB.json` (stage B) and `comparison.json` (combined), produced by `pfx_score.py`,
`pokeflex_stageb/score_stageb.py` and `pokeflex_stageb/compare.py`.

No PokeFlex data, prediction or video is in this repository: the dataset licence is non-commercial with no
redistribution. Only scores and hashes are committed.

## Provenance and blindness

| Step | When (UTC) | Record |
|---|---|---|
| Stage-B blind predictions (PhysTwin, PAC-NeRF) and manifest committed | 2026-11-15 04:08 | commit 5277bc1d, `stageB-MANIFEST.sha256.json` (668 files) |
| v01 code committed (model, baselines, scorer, retry pass) | 2026-11-16 02:40 | commit fa1a5c9c |
| v01 freeze (opens the held-out gate) | 2026-11-16 02:40 | `prediction-freeze.json` (1328 files: ours, fem_lsq, hooke, null) |

- **v01 methods:** scored only if the file is recorded in the freeze and `pokeflex.py verify` passes.
- **Stage B:** scored with the unchanged `pfx_score.score_poke`. Before scoring, two checks run: the manifest must be byte-identical to its version at 5277bc1d, and every file's sha256 must match it.
- **Reporting:** stage B is reported separately, as the protocol amendment requires. It sits in the same tables only so the numbers can be read side by side.

## Coverage, retries and the model limit

- **Simulations:** ours and fem_lsq need one quasi-static FEM run per (poke, ν), with ν in {0, .1, .2, .3, .4}. 762 of 855 runs completed.
- **Retry pass:** first-pass failures that ended in `SimTimeout` were run a second time with a 90-minute budget. Calibration pokes went first, then pokes with the fewest successful ν. This changes compute only: model, data and likelihood are unchanged. The pass rescued 102 runs; 93 failed twice.
- **Why runs fail:** this is not slowness. At extreme compression, some elements reach det F ≈ 0.05–0.08 and Newton stops converging (residual 187 against a 3e-4 tolerance). Bisection then uses up the whole budget. This happens at ν > 0 on foams pressed about 9 cm. Real open-cell foam densifies at that strain, which a compressible neo-Hookean solid cannot represent. This is a limit of the model.
- **Effect on calibration:**
  - FoamDice's calibration poke succeeded only at ν = 0, so its ν posterior is all on 0.
  - 3dPrintedCylinder lacks ν = 0.4.
  - The other six objects have all five ν.
- **Fallbacks for held-out pokes with partial ν coverage:**
  - If no simulated ν has posterior mass, ours applies its E-marginal posterior uniformly over the ν that did simulate (2 pokes: FoamDice_T7-p3, 3dPrintedCylinder_T5-p3).
  - If the fitted ν failed or snapped, fem_lsq uses the nearest ν that did simulate (24 pokes). The note is in each prediction's json.
- **No prediction:** 3dPrintedCylinder_T2-p3 and Pillow_T5-p3 failed at every ν, so neither ours nor fem_lsq has a prediction for them. Ours therefore covers 161 of 163 held-out pokes; hooke, null, PhysTwin and PAC-NeRF cover all 163. Every table below uses the 161 pokes that all methods predicted.

## Force and surface error (mean over held-out pokes per object; 161 common pokes)

Force RMSE, 3-vector (N):

| Object | ours | fem_lsq | hooke | null | PhysTwin (B) | PAC-NeRF (B) |
|---|---|---|---|---|---|---|
| FoamDice | 7.27 | 7.26 | 9.50 | 33.89 | 7.76 | 21.00 |
| FoamCylinder | 3.47 | 3.56 | 4.43 | 6.59 | **3.12** | 4.37 |
| Sponge | 11.63 | **11.46** | 16.39 | 24.62 | 11.63 | 15.41 |
| MemoryFoam | 2.15 | 2.14 | 2.36 | 4.00 | **1.78** | 1.98 |
| FoamHalfSphere | 10.54 | 10.54 | 11.86 | 32.77 | **9.20** | 12.59 |
| Pillow | **12.64** | **12.64** | 13.76 | 22.54 | 14.09 | 17.88 |
| 3dPrintedCylinder | 14.37 | 14.33 | 11.90 | 22.16 | **10.82** | 16.68 |
| 3dPrintedPyramid | 20.14 | 19.62 | 18.21 | 18.61 | 14.52 | **13.03** |
| mean over objects | 10.28 | 10.19 | 11.05 | 20.65 | **9.12** | 12.87 |

Force RMSE along −approach (N), mean over objects:

| ours | fem_lsq | hooke | null | PhysTwin | PAC-NeRF |
|---|---|---|---|---|---|
| 8.39 | 8.29 | 9.11 | 19.82 | **7.49** | 11.73 |

Surface error, tool-excluded symmetric chamfer (mm), mean over objects:

| ours | fem_lsq | hooke/null (rest mesh) | PhysTwin | PAC-NeRF |
|---|---|---|---|---|
| 2.71 | 2.72 | 3.04 | **2.68** | 8.55 |

Per-poke comparison of ours against each method (pokes where ours is better vs worse; median difference, ours minus other):

| vs | force RMSE | force along | chamfer |
|---|---|---|---|
| fem_lsq | 82–78 (0.00 N) | 76–84 (0.00 N) | 53–60 (0.00 mm) |
| hooke | 111–50 (−0.94 N) | 95–66 (−0.29 N) | 149–12 (−0.19 mm) |
| null | 148–12 (−8.2 N) | 150–10 (−9.3 N) | 149–12 (−0.19 mm) |
| PhysTwin | **52–108 (+0.71 N)** | 61–99 (+0.41 N) | 109–52 (−0.04 mm) |
| PAC-NeRF | 112–49 (−2.3 N) | 114–47 (−3.5 N) | 159–2 (−4.9 mm) |

## Stiffness ranking (Spearman correlation with PokeFlex stiffness_N_per_m, 8 objects, 28 pairs)

| Method | scalar | Spearman | discordant pairs |
|---|---|---|---|
| ours | posterior-mean dF/dd of calibration loading | 0.905 | 3 |
| hooke | k | 0.929 | 2 |
| PhysTwin (B) | dF/dd | 0.976 | 1 |
| PAC-NeRF (B) | dF/dd | 0.976 | 1 |

All methods, including ours, fit the catalogue ranking closely. PokeFlex's catalogue values are themselves Hooke/RANSAC fits, which favours spring-like scalars.

## Risk (ours)

- **What it predicts:** P(force RMSE > 2 N) for each held-out poke. The mean predicted risk is 0.89 against an observed rate of 0.94, with a Brier score of 0.061.
- **By band:**
  - 144 pokes had a prediction of at least 0.9; 99.3% of them exceeded the threshold.
  - 17 had a prediction below 0.5; 53% of them exceeded it.
- **Reading:** at the protocol's 2 N threshold, almost every poke exceeds it. The risk score is mainly a sanity check here and is not very discriminative.

## Findings

1. **Ours vs the protocol baselines.**
   - Ours is far better than null and better than a linear spring (hooke): on 111 of 161 pokes for force and 149 of 161 for surface.
   - The exceptions are the 3D-printed cylinder and the pyramid. The spring wins on force there, consistent with lattice-infill parts that do not behave like a homogeneous neo-Hookean solid.
2. **The Bayesian layer adds nothing to the mean prediction here.** ours and fem_lsq are indistinguishable (82–78 on force). The E posterior is very tight (sd log10 E of 0.007–0.04), so the posterior mean is effectively the point estimate. The posterior only adds the risk output.
3. **Stage B: PhysTwin beats ours on force.** It wins on 108 of 161 pokes, by a median of 0.7 N, and has the best object mean (9.1 vs 10.3 N). Surface error is level (2.68 vs 2.71 mm).
   - PhysTwin's spring–mass model has a dashpot, drag and a free collision model. It can represent hysteresis and the densification our elastic model lacks.
   - Its biggest gains are on the 3D-printed objects and FoamHalfSphere.
   - Ours is better on FoamDice and Pillow.
4. **PAC-NeRF is worse than ours on force (112–49) and much worse on surface (159–2, +4.9 mm).**
   - Its rest shape comes from a NeRF reconstruction, with 5–17 mm chamfer to the scan.
   - Its ν sits at the 0.4 bound for 7 of 8 objects.
   - It is the best method on the pyramid, where its fitted shape is good (4.3 mm).
5. **The force errors all sit at the same level.** Every method is at 2–20 N, with the 3-vector RMSE dominated by the along-approach component. No method models unloading hysteresis, which is visible in the comparison videos: measured force drops faster than predicted after peak.

## Caveats

- **Stage-B agent compute:** the agent used the display GPU (D) and up to about 8 processes, beyond the 4 threads it was given (see `stageB-adapters.md` §5).
- **Sealed take manifests:** the agent amended them to register rest-frame images for PAC-NeRF. This happened before the freeze, which hashes the manifests as they are now.
- **PhysTwin support:** bonded support. A first run with ground collision let objects slide or tip and was withdrawn (`runs/pokeflex-v01/stageB/old/`).
- **PhysTwin on the 3dPrintedPyramid:** a poor calibration fit (17.9 N against null 27.0). It is scored as is.
- **Failed simulations:** the 93 twice-failed runs are a model limit (no densification), not a solver budget.

## Videos (local only)

The videos are in `runs/pokeflex-v01/videos/`, git-ignored. There is one per object, for the held-out poke with the median ours force error.

- **Top row:** side-view surface points for measured, ours, fem_lsq, PhysTwin and PAC-NeRF, each coloured by distance to the measured mesh.
- **Bottom row:** force along −approach for each method.
- **To regenerate:** `python -I scripts/pokeflex_stageb/videos.py [--all]`.

These videos are derived from licensed data and must not be committed or shared.
