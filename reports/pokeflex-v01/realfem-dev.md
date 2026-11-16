# pokeflex-v01 — forward model of "ours" (`scripts/pokeflex_v01/realfem.py`): development report

Development take only (FoamDice/FoamDice_T1, protocol `development_takes`). Aggregate numbers only; all
per-frame outputs are under `runs/pokeflex-v01/dev/realfem/` (git-ignored). Model choices follow
`protocol.json` amendment 2026-11-14 `methods_v01.ours`; implementation details that the amendment leaves
open are listed in section 2 so they can be frozen with the code.

## 1. API

- `build(obj, take, poke, h=None) -> RealMesh`: voxel tet mesh of the rest scan at the poke's start frame (free input).
- `simulate(mesh, obj, take, poke, nu, E=None, frames=None, gravity=True)` returns `frames`, `force` (F,3; N, on the tool,
  world frame), `surf_V` (F,Nv,3; m; the scan's own vertices, deformed), `surf_F`, `converged`, `path_ok`,
  `newton_iters`, `substeps`, `jmin`, `n_active`, `residual`, `gravity_sag_max_mm`, `seconds`. Results are deterministic:
  a repeat run with the same thread count matched to 1e-9 N.
- `E=None` runs at `E_REF = 3e4 Pa`. **This differs from the requested `E=1.0` default.** With gravity on, E = 1 Pa
  is unworkable: ρgH/E ≈ 60, so the object collapses under its own weight. Force scales with E (section 5), so one
  run per ν at E_REF, rescaled, covers the E grid.
- CLI: `python -I realfem.py dev [--part mesh|sims|escale|report|all] [--nu ..] [--pokes ..]` runs the development checks;
  `python -I realfem.py meshcheck` builds every poke of every protocol object from free inputs only.

## 2. Model as implemented (choices the amendment leaves open)

- **Geometry.** Cell size h = cbrt(V_scan / 8000), rounded to 0.5 mm and clipped to 5–15 mm. A cell is kept if its
  centre is above y_s (= the minimum scan y) and an upward ray from the centre crosses the scan an odd number of
  times. Three rays offset by about 10 µm vote, which guards against rays that hit an edge exactly. The largest
  face-connected component is kept. Each hex becomes 6 Kuhn tets, which gives conforming faces.
  - The FoamDice rest scans consist of 3–5 patches with duplicated seam vertices. After welding they are closed and
    manifold, including the bottom, so the "open bottom" case did not occur on this take. The upward-ray rule handles
    both cases.
- **Embedding.** Each scan vertex takes the barycentric weights of the tet that contains it, or else of the best
  nearby tet (extrapolated). About 45–50 % of the scan vertices lie outside the staircase. The largest extrapolation
  distance is 6–9 mm at h = 7.5 mm (Pillow: up to 21 mm at h = 13.5 mm).
- **Support.** The bottom grid node layer (y = y_s) is fixed in all directions (bonded). This is how "nodes within
  one cell of y_s" was read; the next node layer, at y_s + h, is free.
- **Gravity.** Gravity is pre-loaded: the solver first finds equilibrium under gravity, then runs the poke. The scan
  is the loaded shape, so `surf_V` = scan + (u − u_gravity). The contact geometry uses the loaded configuration. At
  E_REF the gravity sag is 0.2 mm (density 41 kg/m³).
- **Contact.**
  - **Contact points.** The penalty acts on all embedded scan vertices plus the grid nodes inside the scan. Using the
    scan vertices (about 3.5 mm spacing) gives a 20 mm wide tool face about 30 contact points at any h.
  - **Penalty stiffness.** κ_p = c_pen·E·A_p/h, with c_pen = 100 and A_p = the vertex area (h² for grid nodes). This
    gives a pressure stiffness of 100·E/h.
  - **Penalty energy.** The energy is C2 at contact onset: cubic over 0.1 mm, then quadratic.
  - **Push-out direction.** A point is pushed out along the cheaper direction (out of the end face or radially). The
    direction the point did not come from is weighted ×100 (`K_SWITCH`). "Came from" means the point's side of the
    tool at the previous equilibrium. With the cheaper-direction rule unweighted (or ×5), points near the rim escaped
    radially under the rim stress concentration. The scan surface then slipped around the tool and the force
    "snapped" from about 75 N to 35 N within one frame (dev pokes 3 and 4 at ν = 0.3–0.4). Weighting ×100 keeps the
    energy continuous and removed most of these events (see section 7).
  - **Friction.** The contact is frictionless.
- **Solver.**
  - **Continuation.** Frames are solved in order, each from the previous solution. Tool motion is sub-stepped to at
    most 3 mm of tip travel while in contact, with the last increment as the predictor.
  - **Newton.** Exact tangent, factored with pardiso (symmetric indefinite, symbolic analysis once per mesh). The
    tangent is shifted if the step is not a descent direction. An Armijo line search on the total potential carries a
    positive-Jacobian guard, so no element ever inverts. Tolerance: |r| ≤ 1e-4·(E h² + m g), about 3e-4 N.
  - **Failures.** A failed sub-step is bisected up to 6 times. At the deepest level the failed iterate is discarded and
    the last equilibrium kept: the state then lags the tool by 0.05 mm. Each frame reports `converged` (its own
    residual met the tolerance) and `path_ok` (every sub-step to it converged).

## 3. Meshing statistics

FoamDice_T1, pokes 1–4: h = 7.5 mm, 8024–8102 cells, about 9.6k nodes, about 48k tets, about 27.9k free dof,
298–305 fixed nodes, no cells removed. The voxel volume is 3.385–3.418 L. For comparison, the welded scan encloses
3.42 L and a 15.5 cm cube would be 3.72 L. The resulting density is 41.0–41.4 kg/m³. A build takes about 2 s.

All protocol objects (223 pokes, free inputs only; `runs/pokeflex-v01/realfem_meshcheck.json`): every poke meshed,
and no disconnected cells were found.

| object | h (mm) | cells | volume (L) | ρ (kg/m³) | fixed nodes |
|---|---|---|---|---|---|
| FoamDice | 7.5 | 7.6–8.1k | 3.19–3.42 | 41–44 | 141–324 |
| FoamCylinder | 7.5 | 7.7–8.4k | 3.24–3.53 | 43–47 | 117–394 |
| Sponge | 6.0 | 7.1–7.6k | 1.54–1.64 | 17–18 | 229–733 |
| MemoryFoam | 5.0 | 4.7–5.3k | 0.58–0.66 | 323–365 | 62–387 |
| FoamHalfSphere | 5.0 | 7.1–8.0k | 0.89–1.00 | 41–46 | 662–749 |
| Pillow | 13–13.5 | 7.1–8.4k | 17.2–20.0 | 49–57 | 283–851 |
| 3dPrintedCylinder | 5.5–6.0 | 7.1–9.1k | 1.50–1.63 | 136–148 | 24–272 |
| 3dPrintedPyramid | 5.0 | 2.9–3.9k | 0.37–0.49 | 98–132 | 577–627 |

Points to check before the full run:

- **MemoryFoam.** The scan volume is about 0.6 L, against 1.04 L for the catalogue box, so ρ ≈ 340 kg/m³.
- **3dPrintedCylinder.** Some takes have only 24 fixed nodes, presumably because it lies on its side on a contact
  line: a weak, bonded line support.
- **Pillow.** At h ≈ 13 mm the dent under the 20 mm tool is resolved by about 1.5 cells. Its scan vertices are
  extrapolated up to 21 mm outside the tets.

## 4. Performance

- **Per simulation.** On the development take (31–53 frames per poke, 4 threads per unit, 5–7 units in parallel), 19
  simulations finished. The median took 320 s and the mean 541 s, with extremes of 198 s and 2344 s. A 20th (poke 4,
  ν = 0.4) was stopped unfinished after about 90 minutes.
  - The tail comes from a few frames near peak load or early unloading, where bisection plus 30-iteration Newton
    attempts used up to about 2900 Newton iterations in one frame.
  - A Newton iteration costs about 0.7 s (28k dof).
  - Build plus gravity takes about 3 s.
  - Final-frame convergence: all 712 frames met the tolerance. 4 frames had a failed intermediate sub-step
    (`path_ok = False`).
- **Full run.** Projected cost for 171 complete pokes × 5 ν (6128 frames × 5): about 14 s per frame at the dev mean,
  so about 125 unit-hours. With 8 units × 4 threads that is about 16 h wall-clock, not counting runs like the stopped
  one.
  - Other objects have different h and depths, so this is an estimate.
  - Before a full run, the tail must be capped: a per-frame Newton budget, maxdepth 4 and a wall-clock limit per
    simulation, with capped frames reported as `path_ok = False`. Not done yet.

## 5. E-scaling check (poke 1, ν = 0.3, gravity on; reference E_REF = 3e4 Pa)

| run | max relative deviation of force/E | max force/E deviation × E_REF | RMS relative displacement deviation (surface) | 99th-percentile vertex deviation | gravity sag |
|---|---|---|---|---|---|
| E = 3e5 | 2.8 % | 1.7 N | 5.5 % | 2.0 mm | 0.02 mm |
| E = 3e3 | 3.6 % | 2.2 N | 6.4 % | 1.9 mm | 1.95 mm |
| E = 3e4, no gravity | 2.8 % | 1.7 N | 5.5 % | 2.0 mm | 0 |

The E = 3e5 and gravity-off runs are almost identical to each other, and both differ from the E_REF run by the same
amount. The deviation at E ≥ E_REF is therefore not gravity: it is path sensitivity of the penalty contact. A
perturbation of order 1e-6 moves a few contact points to a neighbouring equilibrium under the tool. The largest
single-vertex deviations (about 11 mm) are under the probe, inside the region the protocol excludes from the
surface score.

Gravity starts to matter at E ≈ 3e3 (1.95 mm sag). Below E ≈ 1e3 Pa the one-run-per-ν rescaling is not reliable.
The fits on this take put E ≈ 2.3–2.6e4, two orders of magnitude above that.

Conclusion: force ∝ E holds to about 3 %, with displacements E-independent to about 5 % RMS. That is the size of
the numerical path noise.

## 6. Fit on the development take

Measured forces are offset-corrected (`P.force_offset`). Scored frames run from start+1 to end. E is fitted in
closed form, pooled over the 4 pokes. The null predictor gives a force RMSE of 33.4 N and an along-axis RMSE of
32.7 N.

| ν | best-fit E (Pa) | force RMSE (N) | along −approach RMSE (N) | per-poke E (Pa) | surface chamfer_excl_tool (mm) | plain chamfer (mm) |
|---|---|---|---|---|---|---|
| null | – | 33.4 | 32.7 | – | 3.51 | 3.84 |
| 0.0 | 22,900 | 6.89 | 5.08 | 21.5k–26.6k | 1.94 | 2.04 |
| 0.1 | 23,400 | 7.05 | 5.52 | 22.0k–27.4k | 2.03 | 2.14 |
| 0.2 | 24,300 | 6.84 | 5.26 | 23.3k–27.8k | 2.12 | 2.22 |
| 0.3 | 25,800 | 6.95 | 5.21 | 24.3k–29.8k\* | 2.29 | 2.39 |
| 0.4 | (24,800, pokes 1–3) | (6.22, pokes 1–3) | (4.93) | – | (2.64, pokes 1–3) | – |

\* At ν = 0.3, poke 4 contains one snap event (section 7). Without it the ν = 0.3 numbers would match the others.

**ν = 0.4 is incomplete.** Its poke-4 simulation was still unfinished after about 90 minutes and was stopped. The ν =
0.4 row therefore covers pokes 1–3 only. On those same three pokes the other ν give:

| ν | force RMSE (N) | along-axis RMSE (N) | surface chamfer_excl_tool (mm) |
|---|---|---|---|
| 0.0 | 7.39 | 5.50 | 2.10 |
| 0.1 | 7.54 | 6.05 | 2.20 |
| 0.2 | 7.36 | 5.89 | 2.30 |
| 0.3 | 7.20 | 5.78 | 2.40 |
| null | 36.7 | – | 3.73 |

How the surface error was computed: mean over 16 frames (4 per poke: mid-loading, deepest, mid-unloading, end), using
the surface at E_REF. At the deepest frames the null predictor scores 7.7–9.0 mm and the FEM 3.4–5.5 mm. At the end
frames the FEM has returned to the rest scan, so its score equals the null's.

**Which ν fits.**

- **Force.** ν cannot be identified from force on this take: all ν between 0.0 and 0.3 give 6.8–7.1 N RMSE, against
  33.4 N for the null. On pokes 1–3, ν = 0.4 is slightly better (6.2 N vs 7.2–7.5 N).
- **Surface.** The tool-excluded surface error prefers small ν (0.0 → 1.94 mm, rising monotonically to 2.29 mm at
  0.3, and 2.64 mm at 0.4 on pokes 1–3).
  The preference comes from the deepest frames (bulging away from the tool).
- **Best overall:** ν = 0.0–0.2, with E ≈ 2.3–2.4e4 Pa. The force error is about 20 % of the null's; the surface error
  is about 55–60 % of the null's.

**Force shape** (see `runs/pokeflex-v01/dev/realfem/force_dev.png`):

- **Onset.** The simulated onset is 1–2 frames late in every poke. The measured force starts rising while the tool
  is still a few mm above the reconstructed rest surface, which could be latency, reconstruction bias or the flat end
  meeting a slanted surface. As a result the measured loading branch lies above the FEM over the first 10–20 mm.
- **Peak.** Peak frame matches to 0–1 frame. Peak magnitude matches to within about 10 % at the pooled E. Poke 1 is
  low (50 vs 57 N); poke 4 is high (65 vs 59 N).
- **Unloading.** Measured unloading leads the FEM by about 1 frame and falls faster. This hysteresis, from the foam's
  rate dependence, is something a quasi-static hyperelastic model cannot produce. The force returns to zero at the
  same frame (off_frame within 0–1).
- **Lateral force.**
  - World Fz is reproduced in pokes 1, 2 and 4: within about 1–2 N at the peaks of −12, −8 and +7.6 N.
  - Fx is not: the measured value is −14 to −16 N in pokes 3 and 4, the FEM gives −3 to −6 N, and the sign is wrong
    in poke 1. The tool slides about 15 mm in +x while pressing, so the measured −x force is most likely face
    friction, which the frictionless contact omits.
  - Lateral RMSE is 2.6–7.0 N per poke and makes up most of the remaining 3-vector RMSE.

## 7. Failure modes and physically questionable points

- **Extreme strain under the probe.** Pokes press 70–75 mm into a 155 mm foam cube with a 10 mm-radius flat punch.
  The smallest element volume ratio reached is J_min = 0.003–0.03 at ν = 0.0 and 0.08–0.21 at ν = 0.4. The neo-Hookean
  ln J barrier keeps elements valid (none inverted), but this is far outside small-strain foam behaviour, and real
  foam densifies. The fitted E is effectively a large-strain secant modulus.
- **Snap / piercing.** Under rim stress concentration, contact points can escape radially, the scan surface slips
  around the tool, and the force drops abruptly.
  - With K_SWITCH = 5: in 3 of 20 dev simulations.
  - With K_SWITCH = 100: in 1 of 20 (poke 4, ν = 0.3, a softened plateau of about 57 N instead of about 80 N, on a
    `path_ok = False` frame).
  - It is a modelling artefact: point-based contact between discrete scan vertices against a sharp rim.
  - Proposed handling: flag simulations with a `path_ok = False` frame followed by a force drop of more than 25 % while
    the tool is still advancing. If it recurs, consider a rounded rim or segment-based contact. Not done.
- **Penalty compliance.** At c_pen = 100 the force is about 6–12 % below the stiff-penalty limit. At c_pen = 300 the
  poke-3 peak was about 10 % higher (an extrapolation from c = 30/100/300 runs). This bias is absorbed into the fitted
  E, so E is biased low by a similar amount. It does not affect the ν-wise comparison.
- **Path sensitivity.** About 3 % of force (section 5) from neighbouring contact equilibria under the tool.
- **Support.**
  - Bonded bottom layer: the object cannot slide or lift. The measured lateral forces of up to 16 N would need a
    friction coefficient of about 0.2–0.3 against the table at about 60–75 N normal load. Whether the real object slid
    was not checked against images.
  - The support plane is the minimum scan y. The cell layer touching the table is staircased (cell centres at
    y_s + h/2).
- **Gravity** is negligible at the fitted E (0.2 mm sag). See section 5 for low E.
- **Not yet exercised.**
  - The ramp-in path for a poke that starts with the tool already penetrating is implemented but was not exercised
    (all dev pokes start out of contact).
  - Objects other than FoamDice were only meshed, never simulated.
