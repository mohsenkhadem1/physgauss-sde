# Figure 1 reference check: COMSOL 6.4 re-solves

**Question.** The Figure-1 reference trajectories ("truth") come from our own nonlinear T10 FEM (`scripts/stem_trial_v01/fem.py`), which is the same FEM family as our method. A reviewer could argue that this biases the truth toward our method. To test this, all three scenarios were solved again in COMSOL Multiphysics 6.4. The COMSOL models use the same continuum physics but a different discretisation: an unstructured mesh, COMSOL's own element assembly and quadrature, a different time integrator, and an exact geometric partition of the load and station regions. The scored quantities were then extracted exactly as `fig1_score.py` defines them.

**Verdict.** The truth is accurate to 0.08-0.25 mm RMS, against a reported method-error scale of 0.3-18 mm. The truth-COMSOL difference is below our own error by these factors: S1 8-23x, S3 2.4-4x, and S2 (the smallest margin) 1.9-4x. In S2 the truth is 0.09-0.16 mm from a COMSOL solution converged in mesh and time, against our 0.30-0.36 mm. Against the best other method the truth-COMSOL difference is at least 7x smaller everywhere, and 9-25x in most cells. The S3 striker impulse agrees to 0.0006-0.0013 N s, about 0.3%. All sealed predictions were re-scored against COMSOL instead of the truth. Station and probe RMS values change by at most 0.07 mm (typically 0.01-0.03 mm), impulse errors change by at most 0.0013 N s, and no ranking changes in any scenario or world. The truth has no detectable bias toward our method.

## Results: truth vs COMSOL in the scored metrics

These are the production COMSOL solutions. All use quadratic tets, BDF rtol 1e-5, and 2-4 cores; the per-case max steps are given in the Setup section.

| Scen. / world | COMSOL mesh (tets) | truth-COMSOL RMS diff (mm) | max diff (mm) | truth signal RMS (mm) | ours, scored (mm) | best other method (mm) | ours / truth-diff |
|---|---|---|---|---|---|---|---|
| S1 W500  | 18 412 (h 13.3 mm) | **0.093** | 0.32 | 30.2 | 2.18 | 8.86 (PhysTwin) | 23x |
| S1 W1000 | 18 412 | **0.084** | 0.27 | 20.1 | 1.01 | 0.79 (M-PhyGs) | 12x |
| S1 W2000 | 18 412 | **0.081** | 0.25 | 15.5 | 0.63 | 2.88 (PAC-NeRF) | 8x |
| S2 Bleft   | 23 181 (h 12.5 mm) | **0.091** | 0.36 | 1.1 | 0.36 | 0.63 (PAC-NeRF) | 4x |
| S2 Bcentre | 65 038 (h 9 mm) | **0.116** | 0.45 | 2.5 | 0.34 | 1.48 (M-PhyGs) | 3x |
| S2 Bright  | 23 255 (h 12.5 mm) | **0.160** | 0.55 | 2.8 | 0.30 | 1.41 (M-PhyGs) | 1.9x |
| S3 W500  | 16 554 (h 13.3 mm) | **0.253** | 0.65 | 12.5 | 0.60 | 6.18 (GIC) | 2.4x |
| S3 W1000 | 16 554 | **0.171** | 0.57 | 13.6 | 0.51 | 1.30 (M-PhyGs) | 3x |
| S3 W2000 | 16 554 | **0.109** | 0.35 | 12.4 | 0.43 | 2.66 (M-PhyGs) | 4x |

In the "best other method" column, the S1 W1000 entry (M-PhyGs, 0.79 mm) is better than ours. The truth-COMSOL difference (0.08 mm) is 10x smaller than that score, so the check confirms this comparison too.

**S3 striker impulse.** The impulse is m (v_end - v0)_z in N s.

| world | truth | COMSOL | truth - COMSOL | ours, abs. err vs truth / vs COMSOL | best other, abs. err vs truth / vs COMSOL |
|---|---|---|---|---|---|
| W500  | 0.5091 | 0.5085 | +0.0006 | 0.0079 / 0.0073 | 0.050 / 0.050 (GIC oracle); best non-oracle 0.123 / 0.123 (PhysTwin) |
| W1000 | 0.3863 | 0.3852 | +0.0011 | 0.0014 / 0.0003 | 0.0030 / 0.0018 (GIC) |
| W2000 | 0.4692 | 0.4679 | +0.0013 | 0.0005 / 0.0008 | 0.0115 / 0.0102 (GIC oracle); best non-oracle 0.053 / 0.052 (M-PhyGs) |

The truth impulse is about 0.3% higher than COMSOL in every world. That is 0.001 N s, the same size as our W1000 and W2000 impulse errors. Our impulse errors are therefore at the truth's resolution limit and should be reported as "<= 0.001-0.002 N s", not as a precise value. The ordering against GIC at W1000 (0.001 vs 0.003 N s) holds against COMSOL as well (0.0003 vs 0.0018 N s). All other impulse errors are 20-1500x larger than the truth uncertainty.

**Re-scoring against COMSOL** (`runs/fig1-v01/comsol/rescore.json`). Every sealed "after poke" and oracle prediction was scored with COMSOL in place of the truth, reading the predictions exactly as `fig1_score.py` does.
* Ours vs truth -> vs COMSOL:
  * S1: 2.18->2.17, 1.01->1.00, 0.63->0.63 mm.
  * S2: 0.36->0.34, 0.34->0.36, 0.30->0.33 mm.
  * S3: 0.60->0.60, 0.51->0.54, 0.43->0.45 mm.
* The largest change for any method is 0.07 mm (GIC, S3 W1000: 7.24->7.30).
* The method ranking is identical in all 9 scenario/world cells.

## Setup (identical for all scenarios unless stated)

* **Code.** COMSOL 6.4 Java API in standalone mode, Solid Mechanics with geometric nonlinearity, 2-4 cores. Sources: `scripts/fig1_v01/comsol/Fig1Comsol.java` (model and evaluation), `run.sh`/`queue.sh` (launch), `compare.py` (truth extraction, comparison, re-scoring). Outputs: `runs/fig1-v01/comsol/` (`<case>.txt` per-frame scored quantities, `<case>.json` setup and timings, `<case>.log`, `<case>.progress` solver logs, `comparison.json`, `rescore.json`, `truth-scored.npz`).
* **Material.** Hyperelastic, user-defined, "Compressible":
  `Ws = mu/2 (J^(2/3) I1_iso - 3) - mu ln J + lam/2 (ln J)^2`.
  This equals the truth's `W = mu/2 (I1-3) - mu ln J + lam/2 (ln J)^2`, i.e. `P = mu(F - F^-T) + lam ln J F^-T`. Lame constants come from E and nu exactly as `fem.lame`. Density is set on the material. There is no damping in either code.
* **Elements.** Free (unstructured Delaunay) tetrahedra with quadratic Lagrange displacement (`order_displacement = 2`). This is the same polynomial space as T10. The mesh is not aligned with the truth's structured 6-tets-per-cube mesh. Quadrature is COMSOL's own.
* **Loads.** All loads are body loads per **reference** volume.
  * Gravity is `-rho g` on all material.
  * Pushes are applied as (total force / exact sub-domain volume) x ramp, with ramp `0.5(1-cos(pi min(t/0.25,1)))`.
  * The load regions and stations are separate geometric sub-domains, so the load and station averages are exact. The truth approximates the same regions with an 84-point lattice indicator inside elements that do not align with the region boundaries. That approximation is part of what this check measures.
  * The integration operators (`intop`) use the material frame. At t_end they return the exact reference volumes (station 1.28e-4 m^3, stem 2.56e-3 m^3, press box 1.6e-5 m^3), as recorded in the logs.
* **Time integration.** BDF variable step, order 1-2 (maxorder 2), adaptive error control with relative tolerance 1e-5 (smoke runs 1e-3), and a maximum step of 1/480 s (S1, S2) or 1/2400 s (S3). Output is interpolated at t = k/48. Newton uses COMSOL's default fully coupled, damped Newton with the direct solver. This integrator is independent of the truth's fixed-step Newmark average-acceleration scheme, which is undamped. BDF2 is slightly dissipative at high frequency, and the error control bounds that effect.

### S1 (stem push D2c)
The free stem x in [-0.20, 0.20] is modelled and the **face x = -0.20 is fixed**. This is equivalent to the truth, which fixes every node with x <= -0.20: the clamp block is rigid and has a mesh plane at x = -0.20. The geometry is partitioned at x = -0.11, -0.09, -0.01, 0.01, 0.085, 0.09, 0.11, 0.115 and 0.18. The 20 N load, direction (0,-16,-12)/20, is applied uniformly over the reference slab 0.085 <= x <= 0.115. Stations are `intop_station(u)/intop_station(1)`, the exact volume averages over the four station slabs used by `ours.STATIONS` and `score.truth_stations`.

### S2 (block press D2)
* Block [-0.12,0.12] x [0.53,0.63] x [-0.06,0.06] with the bottom face fixed.
* The inclusion (|x-xc| < 0.04, y > 0.555, full depth) is a separate domain with 8 mu and 8 lam. This is identical to the truth's centroid assignment, because the truth mesh is aligned to the inclusion boundaries.
* The press box [x-0.02,x+0.02] x [0.62,0.63] x [-0.02,0.02] at x = -0.07 is a separate domain carrying 30 N down x ramp.
* The 27 probes are evaluated by interpolating u at the reference points on the top face (`Interp` numerical feature).

### S3 (stem striker D3)
* The stem is modelled as in S1 (partitioned at the stations and at x = 0.12), with gravity.
* The rigid sphere has **3 translational global ODE states** (cx, cy, cz). This matches `contact.py`: translation only, no sphere gravity, and no rotation. A radial penalty exerts no torque about the sphere centre in any case. Initial conditions are c0 = (0.17, 0.60, 0.0755) and v0 = (0, 0, -1).
* Contact uses the exact `contact.py` law. The force density per reference volume is `b = kappa max(r - d, 0) (x - c)/d` with x = X + u, d = |x - c|, kappa = 1e11 and r = 0.03. It is applied as a body load on the whole stem. The sphere equation is `m c'' + integral(b dV_ref) = 0`, with m = 0.3. The coupled system is monolithic and was stable without special treatment.
* Impulse = m (dcz/dt(0.75) - v0_z).

### Physics differences that remain
None intended. Only the discretisation differs:
1. The mesh, and how it resolves the load and station regions (exact partition vs. lattice indicator).
2. The quadrature of the penalty contact (COMSOL Gauss points on its own mesh vs. the 84-point lattice per T10 element).
3. The time integrator (adaptive BDF2 vs. fixed-step Newmark).
4. COMSOL models the stem from x = -0.20 rather than including the rigid clamp block. This is exactly equivalent.

## COMSOL self-convergence

| check | change in scored quantity (RMS, mm) | each level vs truth (mm) |
|---|---|---|
| S1 W1000, mesh h20 (8 153 tets, rtol 1e-3) -> h13.3 (18 412, rtol 1e-5) | 0.20 | 0.19 -> 0.084 |
| S1 W1000, time rtol 1e-4 -> 1e-5 (same h13.3 mesh) | 0.094 | 0.063 / 0.084 |
| S2 Bcentre, mesh h20 (5 669, rtol 1e-3) -> h12.5 (23 576, rtol 1e-5) | 0.062 | 0.076 -> 0.122 |
| S2 Bcentre, mesh h12.5 -> **h9** (65 038 tets), rtol 1e-5 | **0.018** | 0.122 -> 0.116 |
| S2 Bcentre, time rtol 1e-5 -> **1e-6** (h12.5) | **0.004** | 0.122 -> 0.124 |
| S3 W1000, h20 with tip h10 (12 712, rtol 1e-3) -> uniform h13.3 (16 554, rtol 1e-5) | 0.19; impulse 0.3853 -> 0.3852 | 0.197 -> 0.171; impulse diff -0.0010 -> -0.0011 |

* **S2** was refined in both mesh and time. COMSOL is converged to about 0.02 mm there, so the remaining 0.12 mm difference from the truth is error in the truth itself.
* **S1** was checked for time tolerance and one mesh level. At rtol 1e-5, COMSOL's own error is about 0.02-0.05 mm, judging from the rtol trend and the S2 behaviour.
* **S3**: the impulse is mesh- and tolerance-insensitive to 1e-4 N s. The station RMS changes by about 0.2 mm between the two levels, which bounds COMSOL's station error at this resolution to a few tenths of a mm. A contact-zone refinement (tip h 5 mm, 84k tets) and cubic-element (p3) runs were started but stopped: at about 5 steps/min they needed 10-45 h on the 4-core budget. Their partial logs are in `runs/fig1-v01/comsol/aborted/`.

All solves were stable. The S3 monolithic penalty plus ODE coupling needed no special treatment: BDF reduces its step to 4-9e-5 s during and after impact, then grows it again.

## Interpretation and caveats

* **Where the truth error comes from.** In S1 and S3 the truth and COMSOL agree to 0.08-0.25 mm over 12-30 mm of motion, which is 1-2% of the signal. In S2 the truth-COMSOL difference grows steadily with time, from about 0.05 mm in the first 12 frames to about 0.16 mm in the last 12. It is largest at the probes farthest from the press (x = ±0.1 m). That pattern is phase drift of the undamped gravity ringing of the soft block, which oscillates at roughly 20-25 Hz with about ±0.6 mm amplitude. The likely sources are the truth's coarse structured (18,8,9) mesh (dispersion), fixed-step Newmark at 1/480 s, and the lattice approximation of the 1 cm press layer, which covers only 80% of the truth's 1.25 cm top element layer. This was not separated further, because the truth's code and outputs were not to be modified.
* **Size relative to the reported errors.** The truth error is 0.08-0.25 mm against reported method errors of 0.3-18 mm.
  * Our margin is largest in S1 (8-23x) and smallest in S2 Bright (1.9x) and S3 W500 (2.4x).
  * Because RMS obeys the triangle inequality, no score can move by more than the truth-COMSOL difference. The re-scoring confirms that the scores actually move by at most 0.07 mm.
  * Against the other methods the margin is at least 7x (S2 Bleft vs PAC-NeRF 0.63 mm) and usually 10-100x.
* **Bias.** Truth error large enough to change a comparison exists only where scores differ by less than about 0.1-0.25 mm, or impulses by less than about 0.001 N s. No reported claim rests on such a gap, with one exception. The W1000/W2000 impulse errors of 0.000-0.001 N s for ours are at the truth's resolution, so they should be stated as "<= 0.001-0.002 N s". Re-scoring against COMSOL does not favour our method: our S2 errors rise slightly for Bcentre and Bright and fall for Bleft, and our S3 W1000 station error rises from 0.51 to 0.54 mm.
* **Physics mismatches.** None remains. The COMSOL models use the same energy, density, Lame parameters, boundary conditions, loads, ramp, contact law and sphere dynamics. Differences come only from discretisation, quadrature and the time integrator.
* **Recommended sentence for the paper.** "An independent re-solve in COMSOL 6.4 (unstructured quadratic tetrahedra, adaptive BDF, exact load and station sub-domains) reproduces the reference scored quantities to 0.08-0.25 mm RMS (S1 <= 0.09, S2 <= 0.16, S3 <= 0.25 mm) and the striker impulse to <= 0.0013 N s. Re-scoring every method against COMSOL changes no ranking."

## Run times
Times are wall-clock COMSOL solve times on the shared 36-core machine, which also ran 14 PokeFlex workers.

| case | run time |
|---|---|
| S1, 4 cores | 21 min (rtol 1e-5) and 7 min (rtol 1e-4) |
| S1 W500 / W2000, 2 cores | 44 min / 31 min |
| S2 h12.5, 4 cores | 46 min |
| S2 Bleft / Bright, 2 cores | 62 min / 67 min |
| S2 h9, 2 cores | 4.4 h |
| S2 rtol 1e-6, 2 cores | 3.0 h |
| S3 W1000, 4 cores | 59 min |
| S3 W2000 / W500, 2 cores | 80 min / 2.3 h |
| Smoke runs | 2-14 min |

Total is about 17 h wall-clock, with two 2-core jobs running in parallel for most of it (CPUQuota 200% each). Per-case times are in the `<case>.json` files.
