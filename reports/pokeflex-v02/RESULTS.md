# PokeFlex v02: mechanism ladder on the v01 objects

v02 adds mechanisms one at a time to the frozen v01 model and asks whether each one improves held-out force
prediction, especially during unloading, without making the surface worse.

**This study is post-hoc and not blind.**
- It was motivated by v01's unloading failure: measured force falls faster after the peak than any v01 model predicts.
- It uses the same 8 objects, calibration windows and 161 held-out pokes as v01. Those held-out outcomes, their aggregate scores and the v01 videos had already been seen.
- What was done to keep it honest:
  - every choice was fixed before any v02 fit (`scripts/pokeflex_v02/protocol_v02.json`, written 09:57 UTC);
  - all fitting used calibration data only;
  - rung selection used calibration evidence only, and was recorded before scoring;
  - every prediction file and the code were hashed and committed before any v02 held-out score (commit 2e67415b, `MANIFEST.sha256.json`, 1352 prediction files);
  - the scorer refuses any file that does not match the committed manifest.
- No code, grid or prior was changed after scoring. Everything in this report's interpretation was written after seeing held-out results.

No PokeFlex data or prediction is committed. Only code, the protocol, the manifest and aggregate/per-poke scores are.

## Rungs (all parameters fitted on the v01 calibration window only; no new FEM)

| Rung | Adds | New parameters (prior) |
|---|---|---|
| M0 | v01 `ours`, frozen. Its sealed predictions are scored as they are. | E (log-uniform 1e2–1e7 Pa), ν ∈ {0, …, 0.4} |
| M1 | Quasi-linear viscoelasticity. One Prony term is applied to the elastic force history of the existing simulations: G(t) = g∞ + g₁e^(−t/τ₁), G(0) = 1, 30 fps clock. Contact is unilateral: when the along-approach force is ≤ 0 the predicted force is 0, so contact can be lost during unloading. The surface is the elastic surface; there is no contact-loss lag. | τ₁ log-uniform 0.03–30 s; g₁ uniform 0–0.95 |
| M2 | M1 plus a series contact compliance k_c under the tool. Body depth is u = d − f/k_c, solved per frame together with the Prony state. The surface is the simulation surface shifted to depth u, so the dent is shallower. | k_c log-uniform 1e2–1e5 N/m |
| M3 | M2 plus a compliant support: the whole object translates by δ_s = f/k_s. The parameters are k_eff = (1/k_c + 1/k_s)⁻¹ and the support fraction r = k_eff/k_s. In force, only k_eff matters; r is identified by the surface alone. | r uniform 0–1 |
| M1b | Pre-registered sensitivity check: M1 with 2 Prony terms. Not a rung of the ladder. | τ₁ < τ₂, g₁ + g₂ ≤ 0.95 |

**Inference** is a grid posterior over all parameters, as in v01: Gaussian force and surface likelihood, with plug-in noise taken from each rung's best fit.
- **Predictions** are posterior means over the highest-mass grid points, up to 0.999 cumulative mass or a cap of 600 points.
- **Risk** is P(force RMSE > 2 N), as in v01.
- **Check of the re-implementation:** the M0 recomputed by the v02 code reproduces the v01 posterior exactly (max |Δ| = 0).
- **Check of the scorer:** re-scoring 5 M0 pokes reproduces the v01 scores exactly.

## Calibration-only rung selection (pre-registered; `selection.json`)

**Rule:** move up a rung only if the summed log evidence improves by more than 3 nats.
- **Noise:** fixed at M0's plug-in values.
- **Tempering:** the likelihood is tempered by n_eff/n, with ρ the lag-1 autocorrelation of M0's calibration residuals.
  - Force: ρ = 0.63–0.92, so n_eff is only 4–23% of n.
  - Surface: ρ = 0.57–0.90.

| Step | Δ log evidence (sum over 8 objects) | Objects in favour | Accepted |
|---|---|---|---|
| M1 vs M0 | +2.66 | 7/8 (not Pillow) | **no** (below 3) |
| M2 vs M1 | −1.61 | 1/8 (Pillow +1.2) | no |
| M3 vs M2 | −2.06 | 1/8 | no |
| M1b vs M1 (sensitivity) | +0.24 | – | – |

- **The pre-registered rule selects M0.**
- Untempered evidence, which is reported but was not used for the decision, favours M1 by +127 nats. M2 and M3 are each slightly below M1 on it (−1.5 and −5.0).
- BIC with n_eff favours M0.
- **The held-out results below show that the tempering was too conservative for M1.** The residual autocorrelation it relied on comes mostly from the very misfit that M1 removes.

## Held-out results (161 common pokes; mean over pokes per object, then mean over objects)

**Primary metric:** unloading-phase force RMSE along −approach (N), over frames after the maximum geometric contact depth.

| Object | M0 | M1 | M2 | M3 | M1b | PhysTwin (B) |
|---|---|---|---|---|---|---|
| FoamDice | 5.26 | 4.93 | 4.78 | **4.77** | 4.95 | 6.44 |
| FoamCylinder | 2.72 | **1.44** | **1.44** | 1.48 | 1.47 | 2.86 |
| Sponge | 14.31 | 7.31 | 7.46 | 7.39 | **7.10** | 10.44 |
| MemoryFoam | 2.24 | **0.74** | **0.74** | 0.77 | **0.74** | 1.48 |
| FoamHalfSphere | 6.11 | **4.41** | 4.47 | 4.46 | **4.41** | 5.15 |
| Pillow | 9.89 | 9.88 | 9.13 | 9.19 | 9.89 | **8.40** |
| 3dPrintedCylinder | 9.52 | 10.78 | 10.83 | 10.87 | 10.91 | **7.93** |
| 3dPrintedPyramid | 13.78 | 11.06 | 11.10 | 11.47 | 11.51 | **9.10** |
| **mean over objects** | 7.98 | 6.32 | **6.24** | 6.30 | 6.37 | 6.47 |

The other metrics, as means over objects:

| Metric | M0 | M1 | M2 | M3 | M1b | PhysTwin |
|---|---|---|---|---|---|---|
| loading-phase force along (N) | 8.32 | 8.43 | 8.33 | 8.35 | 8.49 | **8.02** |
| hysteresis-area error (mJ) | 440 | **228** | 232 | 228 | 229 | 278 |
| force RMSE, 3-vector (protocol) | 10.28 | 9.70 | 9.63 | 9.67 | 9.76 | **9.12** |
| force RMSE along −approach (protocol) | 8.39 | 7.67 | 7.59 | 7.62 | 7.73 | **7.49** |
| chamfer_excl_tool (mm, protocol primary) | 2.71 | 2.71 | 2.72 | 2.72 | 2.71 | **2.68** |
| chamfer, no exclusion (mm) | 2.79 | 2.80 | 2.80 | 2.80 | 2.80 | **2.77** |
| tool-region surface (mm) | 7.36 | 7.39 | 7.39 | 7.33 | 7.40 | **7.09** |
| contact-frame force RMSE (N) | 13.57 | 12.77 | 12.70 | 12.74 | 12.85 | **11.99** |
| contact-frame force along (N) | 11.01 | 10.02 | 9.92 | 9.97 | 10.10 | **9.80** |
| peak-force error (N) | 14.45 | **13.94** | 14.29 | 14.29 | 14.02 | 14.75 |
| risk Brier score (P(RMSE > 2 N)) | **0.061** | 0.112 | 0.112 | 0.118 | 0.112 | – |

### Paired per-poke comparisons

Each cell shows the pokes where the first method is better vs worse, then the mean difference (first minus second) with its 95% bootstrap CI. Pokes are resampled in clusters by take, 10,000 times.

| Comparison | unloading force (N) | hysteresis (mJ) | force RMSE (N) | chamfer_excl_tool (mm) | tool region (mm) |
|---|---|---|---|---|---|
| M1 vs M0 | **109–51, −1.56 [−2.39, −0.76]** | 130–30, −210 [−289, −135] | 123–37, −0.55 [−0.86, −0.29] | 63–51, +0.006 [−0.003, +0.017] | 45–68, +0.03 [−0.01, +0.08] |
| M2 vs M0 | 117–43, −1.64 [−2.45, −0.85] | 122–38, −206 [−288, −126] | 119–41, −0.62 [−0.94, −0.34] | 81–79, +0.011 [+0.000, +0.023] | 110–49, +0.04 [−0.22, +0.32] |
| M3 vs M0 | 116–44, −1.58 [−2.40, −0.79] | 122–38, −210 [−291, −132] | 117–43, −0.59 [−0.89, −0.32] | 65–95, +0.018 [+0.000, +0.036] | 109–50, −0.03 [−0.28, +0.21] |
| M2 vs M1 | 75–85, −0.08 [−0.20, +0.01] | 77–83, +4 [−11, +19] | 81–79, −0.07 [−0.20, +0.04] | 77–83, +0.005 [−0.002, +0.012] | 126–33, +0.00 [−0.25, +0.29] |
| M3 vs M2 | 71–89, +0.06 [+0.02, +0.11] | 85–75, −4 [−8, −0.5] | 83–77, +0.03 [+0.01, +0.07] | 72–88, +0.007 [−0.006, +0.020] | 77–82, −0.07 [−0.15, −0.004] |
| M1b vs M1 | 72–88, +0.06 [+0.00, +0.12] | 78–82, +0.3 [−6, +6] | 70–90, +0.06 [+0.02, +0.10] | 24–30, −0.001 | 15–38, +0.004 |
| M0 vs PhysTwin | 54–106, +1.54 [+0.79, +2.30] | 34–126, +168 [+73, +259] | 52–108, +1.25 [+0.40, +2.14] | 109–52, +0.06 [−0.18, +0.41] | 80–80, +0.26 [−0.20, +0.78] |
| M1 vs PhysTwin | 97–63, −0.02 [−0.76, +0.77] | 96–64, −42 [−117, +26] | 83–77, +0.70 [−0.14, +1.57] | 105–56, +0.07 [−0.18, +0.41] | 75–85, +0.30 [−0.17, +0.82] |
| M2 vs PhysTwin | 96–64, −0.10 [−0.82, +0.66] | 96–64, −38 [−113, +31] | 88–72, +0.63 [−0.20, +1.49] | 103–58, +0.07 | 83–77, +0.30 |

### Pre-registered criteria (`scores.json` → `criteria`)

| Rung | Unloading improves vs M0 (CI < 0) | Surface: mean-over-objects Δ ≤ 0.05 mm | Surface: CI upper ≤ 0.10 mm | No degradation |
|---|---|---|---|---|
| M1 | yes | +0.006 | 0.017 | yes |
| M2 | yes | +0.011 | 0.023 | yes |
| M3 | yes | +0.018 | 0.036 | yes |
| M1b | yes | +0.006 | 0.016 | yes |

## Fitted parameters and identifiability (`fit-summary.json`; posterior mean ± sd)

| Object | M0 log₁₀E | M1 log₁₀E (instantaneous) | M1 log₁₀E∞ (relaxed) | g₁ | log₁₀τ₁ (s) | M2 log₁₀k_c | M3 r (sd/prior sd) | M3 corr(log k_c, log k_s) |
|---|---|---|---|---|---|---|---|---|
| FoamDice | 4.35 | 4.63 ± 0.04 | 4.34 | 0.48 ± 0.05 | −1.50 ± 0.07 (prior edge) | 4.4 ± 0.4 | 0.49 (0.96) | +0.17 |
| FoamCylinder | 3.77 | 4.20 ± 0.11 | 3.49 | 0.80 ± 0.04 | −1.01 ± 0.25 | 4.3 ± 0.6 | 0.45 (0.99) | +0.29 |
| Sponge | 4.49 | 4.71 ± 0.07 | 4.19 | 0.66 ± 0.14 | −0.72 ± 0.37 | 4.9 ± 0.2 | 0.45 (0.99) | −0.53 |
| MemoryFoam | 3.87 | 4.33 ± 0.10 | 3.64 | 0.79 ± 0.06 | −1.28 ± 0.23 | 4.0 ± 0.7 | 0.45 (0.99) | +0.49 |
| FoamHalfSphere | 4.65 | 4.87 ± 0.04 | 4.64 | 0.41 ± 0.06 | −1.46 ± 0.13 (near edge) | 4.9 ± 0.2 | 0.47 (0.99) | −0.48 |
| Pillow | 4.14 | 4.15 ± 0.01 | 3.91 | 0.34 ± 0.27 | +0.97 ± 0.57 | **3.19 ± 0.15** | **0.10 (0.30)** | +0.05 |
| 3dPrintedCylinder | 4.12 | 4.21 ± 0.03 | 3.71 | 0.62 ± 0.20 | −0.30 ± 0.31 | 4.7 ± 0.3 | 0.56 (0.98) | +0.12 |
| 3dPrintedPyramid | 4.91 | 4.98 ± 0.08 | 4.65 | 0.46 ± 0.25 | −0.24 ± 0.85 | 4.4 ± 0.5 | 0.44 (0.99) | +0.11 |

### Plausibility

- **The Prony term is a fast stiff transient, not slow creep.** For the foams, τ₁ is 0.03–0.2 s, and on FoamDice and FoamHalfSphere it sits at or near the prior's lower edge (one frame). g₁ is 0.4–0.8, so the instantaneous modulus is about 2–5 times the relaxed one. The relaxed modulus E∞ comes back close to v01's E on FoamDice and FoamHalfSphere.
  - In effect M1 acts like a rate-dependent dashpot.
  - Real foams, memory foam especially, relax over seconds. With only one roughly 1 s calibration poke at a single speed, τ cannot be separated from the poke speed. The Prony term may also be absorbing other dissipation (friction, contact-loss dynamics).
- **Pillow** has the only slow τ (about 9 s, weakly identified). M1 does nothing there.
- **3dPrintedCylinder** gets worse with M1 on unloading (9.5 → 10.8 N). Its lattice parts are not a homogeneous viscoelastic solid.
- **Sponge** gets better on unloading (14.3 → 7.3) but worse on loading (7.0 → 10.4). This fits a densifying material that the QLV approximates by stiffening the instantaneous response.
- **Contact compliance k_c** is ≥ 1e4 N/m on 7 of 8 objects, i.e. negligible next to the objects' 250–1250 N/m. The exception is Pillow, at k_c ≈ 1.5e3 N/m, where a soft fabric/fill layer is plausible. On Pillow, M2 is the only rung with positive evidence (+1.2 nats) and it improves held-out unloading (9.9 → 9.1) and force RMSE (12.6 → 12.0).

### Is M2 separable from M3?

**No.** The surface term does not identify the split between contact compliance and support compliance.
- On 7 of 8 objects the posterior of r is essentially the prior (sd/prior sd 0.96–0.99), and corr(log k_c, log k_s) ranges from −0.53 to +0.49.
- **Why:** a rigid translation of several mm barely changes the tool-excluded chamfer. On the dev check, a 5 mm translation changed it by 0.004 mm, against about 2.5 mm of per-frame surface noise.
- **The one exception is Pillow:** r = 0.10 ± 0.09 (sd/prior sd 0.30). Its compliance is under the tool, not in the support.
- **Held-out:**
  - M3 is slightly worse than M2 on force (+0.03 N, CI excludes 0).
  - M3 is slightly better on the tool region (−0.07 mm, CI excludes 0).
  - Both effects are negligible.

## Answers

1. **Rate dependence (M1) works on held-out data.**
   - Unloading force error falls by 1.56 N, about 20% (CI −2.39 to −0.76), better on 109 of 161 pokes.
   - Hysteresis-area error roughly halves.
   - Protocol force RMSE improves by 0.55 N (123–37).
   - Surface is unchanged (+0.006 mm, CI upper 0.017).
   - On unloading, M1 draws level with PhysTwin, and the CI includes 0. On whole-poke force it is still 0.7 N behind (CI −0.14 to +1.57), mainly because of the 3D-printed objects.
   - **However, the pre-registered calibration rule did not select it** (+2.66 against the 3-nat bar). The tempering was conservative, and that verdict is now known to be wrong for M1. Any change to the rule is post-hoc.
2. **Contact compliance (M2) adds nothing over M1 overall** (−0.08 N unloading, CI −0.20 to +0.01). The exception is Pillow, where it helps.
3. **Compliant support (M3) adds nothing and is not identifiable** from force or from the tool-excluded surface.
4. **M1b (2 Prony terms) is slightly worse than M1** (+0.06 N, CI excludes 0). One term is enough for one calibration poke.
5. **The risk output got worse** (Brier 0.061 → 0.112). M1's predicted risk (0.78) is now below the observed rate (0.88).

## Recommendation

- **Do not escalate to full spatial Gaussian-patch FEM.**
  - The reduced M2 and M3 were not accepted on calibration evidence and did not improve held-out over M1.
  - The surface data cannot separate support from contact compliance, so spatial patches would not be identifiable from this data either.
  - No FEM re-simulations were run.
- **Carry M1 (QLV with unilateral contact) forward as the v03 candidate**, but confirm it prospectively, because this v02 is non-blind. Options are new PokeFlex objects or takes, or a sealed re-split. Before that run, fix:
  - the selection rule: calibration CV across pokes, or a block likelihood, instead of AR(1)-tempered evidence calibrated on M0 residuals;
  - the τ prior's lower bound, or add an explicit dashpot term;
  - the risk calibration.
- **The remaining errors are on loading**, not on mechanisms in this ladder:
  - force onset precedes geometric contact on several objects (rest-scan offset);
  - densification on Sponge and FoamHalfSphere;
  - the lattice 3D-printed parts.

## Compute

- **Total:** about 12.5 CPU-h on at most 8 single-thread capped units (`pfx2-*`). No GPU and no new FEM; v02 reuses v01's existing simulations.
- **Breakdown:**

| Step | CPU time | Detail |
|---|---|---|
| Surface tables | 4.7 h | 21,608 chamfers |
| Fits and predictions | about 0.4 h | 8 objects × 4 rungs |
| Scoring | 7.3 h | 4 rungs × 161 pokes, protocol and contact metrics on every frame |

## Files

- `scripts/pokeflex_v02/`:
  - `protocol_v02.json`;
  - `v02_model.py` (forward model);
  - `v02_fit.py` (surface tables, fit, select, predict);
  - `v02_manifest.py`;
  - `v02_score.py` (gated scorer);
  - `v02_params.py`;
  - `run_v02.sh`.
- `reports/pokeflex-v02/`:
  - `MANIFEST.sha256.json` (committed before scoring);
  - `selection.json`;
  - `fit-summary.json`;
  - `scores.json` (aggregates, paired comparisons, criteria, per-poke score rows).
- Predictions in `runs/pokeflex-v02/pred/` are git-ignored and licensed: do not share them.
