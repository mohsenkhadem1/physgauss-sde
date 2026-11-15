# PokeFlex stage B: PhysTwin and PAC-NeRF adapters (blind predictions)

Status: blind predictions for the held-out pokes, written before the v01 freeze. Stage B is reported on its own
and is not part of the v01 freeze (protocol amendment `stage_b_not_in_v01`). Nothing has been scored. No held-out
outcome was opened. Code: `scripts/pokeflex_stageb/`. Predictions: `runs/pokeflex-v01/pred-stageB/<method>/`.
Manifest: `runs/pokeflex-v01/pred-stageB/MANIFEST.sha256.json`.

<!--RESULTS-->

## 1. Blindness and data access

* Every PokeFlex read goes through the gated loader (`pokeflex.py`). It runs in two CPU export steps, each with `python -I`:
  * `prep_inputs.py` exports, per held-out poke, only free inputs: the rest scan at the start frame, the tool
    trajectory `T_WT` for frames start..end, the contact depth from free inputs, and the approach axis. For the
    calibration poke it also exports the calibration-window outcomes: offset-corrected forces
    (`pfx_common.calibration_force`), and 4000 area-weighted samples of each calibration-frame scan with the protocol's
    tool-exclusion mask. Calibration pokes that start in contact are dropped, using the same rule as `pfx_ours.fit_object`
    (FoamDice_T2 p1, Sponge_T1 p1, Pillow_T1 p1). One calibration poke per object remains.
  * `pac_prep.py` reads the colour images of REST frames only, plus the camera parameters.
  * The GPU adapters (`pfx_phystwin.py`, `pfx_pacnerf.py`) never import the loader. They read only these exports.
* Images: `extract_rest_images.py` runs the protocol's own extractor (`extract.py`) with a member filter. It extracts
  colour images of the start frame of every poke segment (free) and of every frame of the development take, nothing
  else. It asserts that no whitelisted frame is in `P.heldout_frames`. Robot/force members are skipped, so
  `robot_traj.json` and `forces.npy` are not rewritten. `extract.py` regenerated each take's sealed
  `MANIFEST.sha256.json` to include these images. This happened before the freeze, which hashes the manifests then.
* The development take FoamDice_T1 was used only to debug the pipeline (camera conventions, NeRF and MPM mechanics,
  PhysTwin timings). It was never used to fit a parameter.
* Nothing was written in `scripts/pokeflex_v01/`, and `prediction-freeze.json` was not touched.
* The catalogue stiffness (`objects.json` stiffness_N_per_m) was not read. The catalogue mass was read for density, as in ours.

## 2. Common conventions (identical to ours)

* Tool: rigid flat-ended cylinder, r = 10 mm, axis = `T_WT` z, end face at the `T_WT` origin, body 192 mm up the
  axis. It is driven kinematically by the recorded trajectory and interpolated linearly within a frame.
* Support plane: y_s = the lowest vertex of the poke's rest scan (world +y up). Gravity is on. Mass = catalogue mass.
* Each held-out poke starts from its own rest state, the free start-frame scan. Only global parameters fitted on the
  calibration poke are re-used.
* Rest scan as the loaded shape: a gravity settle phase (0.5 s PhysTwin, 0.33 s PAC-NeRF) runs with the tool at its
  start pose. Reported geometry = rest + (x - x_settled), the same convention as ours.
* Force: the force ON THE TOOL in the world frame, N, averaged over the last quarter of each frame's substeps. It is
  zero out of contact. Prediction format = `pfx_common.save_pred` (frames start+1..end, force (K,3), surf_V (K,N,3),
  surf_F; no risk), written by `sb_common.save_pred` rooted at `pred-stageB`.
* Calibration loss, the same for both adapters: force RMSE [N, 3-vector, all calibration frames] + RMS chamfer [mm]
  (observed tool-excluded calibration-scan samples to the predicted points). This puts the protocol's two scored
  units on a 1 N = 1 mm scale.
  * A first version normalised each term by its null value (mean squared). On the calibration data it made the
    chamfer term dominate. For FoamHalfSphere it picked a near-zero-stiffness PhysTwin (36 N force RMSE).
  * It was replaced before any held-out prediction was kept. The PhysTwin predictions from the first version were
    deleted.
* Stiffness-ranking scalar (both methods): the least-squares slope of the predicted calibration push force along
  -approach vs contact depth over the loading half (`sb_common.loading_slope`, as ours' dF/dd). Calibration data only.

## 3. PhysTwin adapter (`ptw_geom.py`, `pfx_phystwin.py`)

The official PhysTwin spring-mass kernels are used unmodified: `eval_springs`, `update_vel_from_force` and
`integrate_ground_collision` from `runs/third_party/PhysTwin-v06`. They run in an explicit substep loop that is
captured as a CUDA graph per frame. Env: `runs/phystwin-env`.

* **Inputs.** Native PhysTwin consumes RGB-D point tracks plus a shape-completed interior. PokeFlex gives
  per-frame reconstructed meshes with no correspondence, so there are no tracks. Following the brief, the adapter
  uses a chamfer fit on the calibration-frame scan samples, with the tool as a kinematic controller.
* **Nodes.** Two kinds of node per poke, both from that poke's rest scan:
  * surface nodes: scan vertices voxel-downsampled at spacing h;
  * interior nodes: h-lattice cell centres inside the scan (the ray-parity vote of realfem._voxelise), more than h/2
    from any surface node.
  h is fixed per object from the calibration rest volume (~12k interior nodes, 4-11.7 mm), so global parameters
  transfer between pokes. Each node has the native unit mass. The physical node mass m = catalogue mass / N, so
  physical forces enter as f/m and physical spring stiffness = Y_nat m.
* **Springs.** The native topology rule: hybrid radius / max-neighbour search, unique pairs. The global stiffness is
  parameterised physically (Y_phys, N per unit strain). The native clamp `spring_Y_max` = 1e5 is lifted, as in the
  Figure-1 adapter.
* **Frame and support.** Native frame: ground z = 0, gravity -z. World maps to native as
  (x, y - y_s, z) -> (x, -z, y - y_s). **Support is bonded, as in ours:** nodes within 0.75 h of the support plane
  are held at rest (a rigid version of PhysTwin's native fixed-controller points). The native ground collision is
  kept for everything else, with collide_elas and collide_fric fitted.
  * A first complete run used the native ground collision alone. Objects slid or tipped under the tool, and on
    3dPrintedPyramid the calibration fit peaked at 5 N against 64 N measured. That run was withdrawn before scoring
    and kept for reference in `runs/pokeflex-v01/stageB/old/` (not in the prediction directory, not in the manifest).
    All 8 objects were refit and all 163 pokes re-predicted with the bonded support.
* **Tool.** A penalty on nodes inside the cylinder. Each node is pushed out through the end face or radially, on the
  side the node came from, which is ours' K_SWITCH rule with K = 100. The penalty stiffness per node is 4 m omega_max^2.
  The substep is dt = min(5e-5, 0.25/omega_max) s, where omega_max is the stiffest node's spring frequency. This adds
  696-22000 substeps per 1/30 s frame.
* **Fit.** Native stage A: CMA-ES (`cma`, sigma 1/6, seed 42, normalised box, 12 generations x 9). It searches over
  log Y_phys (box +-0.5 decade), radius factor (1.2-3 h), max_neighbours (10-50), collide_elas, collide_fric, dashpot
  (0-300) and drag (0-10). The box centre comes from a coarse stiffness scan at the native defaults:
  * FoamDice, FoamCylinder and Sponge used a 0.25-decade line search from the Hooke-slope guess Y ~ k h;
  * the other objects used a 0.5-decade grid from 3 decades below the guess, then +-0.25 decade.
  The scan was made robust after FoamHalfSphere's line search stopped early at a numerically exploding stiffness.
  Explosions are kept as part of the method; CMA sees them as a large loss.
* **Native stage B not run.** The per-spring Adam stage was skipped because each held-out poke builds new springs
  from its own rest scan. A per-spring field fitted on the calibration poke cannot be transferred: the object is
  re-placed 3-35 cm between takes and there is no correspondence.
* **Surface.** Rest-scan vertices moved by an inverse-distance blend of their 4 nearest nodes' displacements (like
  PhysTwin's kNN interpolation of Gaussians). The scan topology is kept. Force = minus the sum of penalty forces.

## 4. PAC-NeRF adapter (`pac_prep.py`, `pfx_pacnerf.py`)

The official PAC-NeRF code is used unmodified (`runs/gap-crossmethod-v02/vendor/pac-nerf`, b613048), run through
`pac_env.sh -I`. The adapter imports and calls it.

* **Images.** The 6 colour cameras: kinect 0/1 (4K), realsense 0/1 (wrist cameras with per-frame extrinsics) and
  volucam 0/1.
  * Conventions were verified on the development take by projecting its scan: extrinsics are world->camera, with
    OpenCV distortion (8-coefficient rational for kinect). Images are `cv2.undistort`ed.
  * PokeFlex ships no masks, and native PAC-NeRF's matting needs a background plate, which PokeFlex lacks. The mask
    is the rest scan's silhouette (free input), dilated 4 px, minus green-screen pixels (chroma key; kept in
    1172/1176 views). The background is set to white, as native.
  * Each view is a square crop around the silhouette (+25 %) resampled to 256 x 256, with intrinsics adjusted. Rays
    follow `dvgo.get_rays_of_a_view(inverse_y=True)`. The result is written as native `data.pt`.
* **Static stage (native `train.train_static`).** Runs on the 6 views of the calibration poke's rest frame.
  * Settings: 1500 iterations, progressive grid upsampling at 250/500/1000, entropy and volume weights 1e-3, batch
    2^14 rays. The bbox is the rest scan's bbox +-15 mm. The final voxel is chosen for ~12k occupied voxels: 4-15 mm,
    6.3 mm for FoamDice.
  * Particles come from native `initialize_particles` (half-voxel lattice). Particles with alpha < 0.05 are dropped:
    they are nearly transparent, with mass ~alpha and stiffness ~alpha^3.
* **Physics.** The native MPM kernels run unmodified: `compute_F_tmp`, `project_F`, `p2g` with the native elasticity
  material (compressible neo-Hookean, mu/lam ~ alpha^3), and `g2p`. The adapter supplies only the grid boundary
  operator:
  * gravity;
  * a sticky support (y <= y_s), as native `ground`;
  * the kinematic tool. Grid nodes inside the cylinder lose the velocity component moving into the tool, relative to
    the tool's velocity. The normal is the tool axis for nodes within 1.5 dx above the end face, otherwise the
    nearer of face and side. Force on the tool = -(momentum removed)/dt.
  The first version used the nearer-surface normal everywhere. Grid nodes deep inside the tool body then pushed
  sideways, giving 2 N vertical force at 97 mm depth on the calibration poke. It was fixed before any held-out
  prediction was kept.
  * Density = catalogue mass / sum(alpha p_vol).
  * Mass scaling x10 with gravity / 10, so the weight stays physical. The pokes are quasi-static (tool speed
    <= ~0.1 m/s), and this gives a 3.2x larger time step: dt = 0.3 dx / c_p.
  * The native rendering-based velocity and dynamic stages are not run, for two reasons. The object starts at rest
    (v0 = 0 is known). And under a displacement-driven poke the images do not constrain E: deformation is
    E-independent up to gravity and inertia, and the reconstruction does not resolve the dent.
* **Fit.** (E, nu) are fitted on the calibration poke with the common loss:
  1. For each nu in {0, 0.2, 0.4}, run one MPM rollout and LSQ-rescale E (force is ~linear in E). The starting E is
     the Hooke-slope guess k/(2R), then the previous nu's rescaled E.
  2. Pick nu by the loss estimated at the rescaled E.
  3. Re-simulate at the chosen (E, nu); that rollout is the reported calibration fit.
  The PAC-NeRF nu bound is +-0.45.
* **Held-out pokes.** A native static stage per held-out poke costs ~5-10 min of contended GPU time with 1500
  iterations (1.1-1.7 it/s with the GPUs shared), ~20 GPU-h for 163 pokes. That does not fit the budget.
  * Instead, each held-out poke re-uses the object's calibration-rest NeRF reconstruction (particles + alphas). It is
    rigidly registered to that poke's free rest scan by multi-start symmetric ICP: 8 yaws x {identity, x-flip,
    z-flip}, centroids aligned, 40 iterations. The ICP residual is recorded per poke.
  * Consequence: PAC-NeRF does not see the residual shape change between pokes. It sees the object's pose only
    through the registration to the scan.
  * The code still supports `PAC_HELDOUT=static`, the per-poke native static stage. Its per-poke data (`data.pt` for
    every held-out rest frame) is already built.
* **Surface.** A marching-cubes surface of the particle occupancy (cell = NeRF voxel, mean alpha, Gaussian
  sigma 0.5 cell, level 0.5), keeping the largest component. It is moved by an inverse-distance blend of the 4
  nearest particles' displacements, so the topology is fixed per poke.

## 5. Results before the freeze (calibration window only; no held-out outcome read)

**Coverage.** Both methods have predictions (force trajectory + surface per frame) for all 163 held-out pokes of
the 8 objects. Manifest: `runs/pokeflex-v01/pred-stageB/MANIFEST.sha256.json`, committed to git before the v01
freeze as the timestamped proof of blindness.

**Calibration-fit quality** (`calib_quality.py`; protocol surface metric on every 2nd calibration frame, force
RMSE over the calibration poke vs the offset-corrected measurement; null = rest scan held fixed, zero force).
Zero-area faces in predicted surfaces are dropped before sampling (17 on the Sponge PAC-NeRF surface).

| object | PhysTwin force N / surface mm | PAC-NeRF force N / surface mm | null force N / surface mm | peak meas. N |
|---|---|---|---|---|
| FoamDice | 9.2 / 1.95 | 25.1 / 10.6 | 41.3 / 4.49 | 78.3 |
| FoamCylinder | 2.2 / 1.24 | 3.0 / 10.0 | 5.5 / 1.61 | 10.9 |
| Sponge | 6.8 / 2.65 | 10.1 / 6.9 | 32.9 / 2.90 | 77.4 |
| MemoryFoam | 1.2 / 0.82 | 1.1 / 5.7 | 2.3 / 0.89 | 5.5 |
| FoamHalfSphere | 5.7 / 1.70 | 17.0 / 6.1 | 37.2 / 1.89 | 86.0 |
| Pillow | 4.8 / 2.73 | 8.6 / 17.3 | 13.2 / 2.88 | 28.8 |
| 3dPrintedCylinder | 4.4 / 2.49 | 13.1 / 8.8 | 16.5 / 3.06 | 37.5 |
| 3dPrintedPyramid | 17.9 / 2.45 | 10.5 / 4.3 | 27.0 / 2.45 | 64.2 |

* **PhysTwin** (bonded) beats null on force on every object and on surface on 7 of 8. **3dPrintedPyramid is a poor
  fit:** force 17.9 N RMS (peak 37 vs 64 N) and a surface error identical to null at every stiffness tried, i.e.
  the tool barely deforms the model and only the spring stiffness moves the force. It is scored, not dropped, and
  flagged as an adapter weakness.
* **PAC-NeRF** beats null on force everywhere but is worse than null on surface everywhere (4.3-17.3 mm vs
  0.9-4.5 mm). The cause is the NeRF reconstruction on real images: rest-shape chamfer 4.5-18.9 mm against the
  scan (ghost volume; the tool meets it up to ~18 mm early), plus the held-out re-use of the calibration
  reconstruction (rigid ICP residual median 4.9-19.1 mm per object). nu sits at the grid edge 0.4 for 7 of 8 objects.

**Fitted parameters and ranking scalars** (`summarize.py` -> `runs/pokeflex-v01/stageB/summary.json`):

| object | PhysTwin dF/dd N/m | PhysTwin fit min | PAC-NeRF E kPa | PAC-NeRF nu | PAC-NeRF dF/dd N/m |
|---|---|---|---|---|---|
| FoamDice | 854 | 22 | 88.9 | 0.4 | 590 |
| FoamCylinder | 172 | 10 | 23.8 | 0.4 | 140 |
| Sponge | 1236 | 82 | 97.6 | 0.4 | 1158 |
| MemoryFoam | 169 | 9 | 9.2 | 0.0 | 188 |
| FoamHalfSphere | 1387 | 81 | 115.5 | 0.4 | 1784 |
| Pillow | 337 | 5 | 48.8 | 0.4 | 255 |
| 3dPrintedCylinder | 438 | 8 | 138.2 | 0.4 | 268 |
| 3dPrintedPyramid | 1110 | 59 | 138.4 | 0.4 | 1675 |

The catalogue stiffness was not read; held-out scores follow the v01 freeze, reported separately from the v01 table.

**Compute.** GPUs A, B and D (display card, 3-6 GB for PAC-NeRF when A/B were saturated); never 67:00.
Up to ~8 adapter processes at once while the CPU FEM workers ran.
