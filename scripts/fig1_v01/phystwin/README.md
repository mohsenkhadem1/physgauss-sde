# PhysTwin adapter (Figure-1 S1/S2/S3)

Runs the official PhysTwin spring-mass digital twin (upstream `81c71879`, isolated copy
`runs/third_party/PhysTwin-v06` = upstream + `scripts/phystwin/controlled-physics.patch` (opt-in `gravity_scale`)
+ `scripts/phystwin/v06-prescribed-forces.patch` (opt-in per-node force file); not modified here) on the Figure-1
scenes. It uses both native optimisation stages and then forecasts the held-out commands. Env: `runs/phystwin-env`.
Outputs: `runs/fig1-v01/phystwin/`.

| file | role |
|---|---|
| `ptw.py` | geometry sampling, force series (rule of `gic_force_adapter._prepare`), native-format dataset, config |
| `fit.py SCENE WORLD EP` | stage A native `OptimizerCMA.optimize` (20 gens), stage B native `InvPhyTrainerWarp.train` (200 it) |
| `forecast.py SCENE WORLD FIT_EP --targets ...` | rollouts: native forward CUDA graph (force commands), substep loop for the striker |
| `forecast.py ... --homog-scan` / `--consistency` | sanity (b) homogeneous scan on D0; substep loop vs native graph |
| `pipeline.sh SCENE WORLD` | D0 fit -> D1 refit (initialised from D0) -> forecasts from both |

Production (one capped unit per world; peak GPU 0.55 GB block / 1.3 GB stem):

    scripts/stem_trial_v01/run_unit.sh ptw-block-Bleft 6G $PWD $PWD/runs/fig1-v01/phystwin/logs/pipe-block-Bleft.log \
        CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 -- scripts/fig1_v01/phystwin/pipeline.sh block Bleft

Outputs: `forecast-{EP}-{D0|D1}-{WORLD}.npz`: `xyz (F,N,3)` all nodes (frame 0 = rest, 48 fps; the trailing `fixed`
nodes are the held boundary), `rest`, `fixed`, `marker_index`. Stem files add `station_centroid (F,4,3)` (nodes binned by
rest x). D3 adds `striker (n_substeps,7)` = (t, c, v). The `.json` holds the fit summary, force audit with the read-back
injected force, wall seconds and GPU. `<scene>/<EP>-<WORLD>/params.json` holds per-stage seconds, GPU peak, CMA optimum,
loss history, and the per-spring stiffness spread (native units = N per unit mass; physical = x node mass).

## Mapping and deviations from native PhysTwin
1. Input: no RGB-D, segmentation, CoTracker, TRELLIS shape completion or Gaussians. The tracked "object_points" are
   the RGB-triangulated markers of the fitting episode (block 20, stem 16), all visible. Native PhysTwin uses dense
   tracked RGB-D surface points. The completed shape is the known rest box, sampled as a cell-centred lattice:
   block 1 cm (2880 nodes), stem 8 mm (5000 free nodes). Markers are extra surface nodes. Surface-layer nodes are
   the chamfer candidate set (`surface_points`), the rest are `interior_points`. No dense interior ground truth is used.
2. Masses: native unit mass per node is kept. Node mass m0 = 1000*V/N is uniform (block 0.99 g, stem 0.51 g), so the
   known force enters as force/m0 (identical dynamics). Physical spring stiffness = native spring_Y * m0.
3. Gravity (-y, 9.8) and the command (ramp sampled at frames, linear within a frame by the native prescribed-force
   interpolation) are in the external-force file. Native z-gravity is off (`gravity_scale=0`), which also removes
   the native z=0 ground plane (no ground contact in these scenes). Collision parameters are then inert, but they
   stay in the 12-dim CMA vector as natively.
4. Boundary: native controller points held still (block: 288 points on the glued face; stem: 500 lattice nodes
   with x <= -0.20), coupled by native controller springs (radius/neighbours searched by CMA).
5. Stem only: `spring_Y_max` 1e5 -> 1e6 (best homogeneous D0 fit sits on the native clamp) and 834 substeps/frame
   (dt 2.5e-5; native dt 5e-5 diverges above spring_Y 5.6e5). Block: native real.yaml (417 substeps, dt 5e-5).
6. FPS 48, train_frame = whole episode (block 25, stem D0 25 / D1 37). Seed 42, CMA sigma 1/6, 20 gens, 200 Adam
   it, lr 1e-3, best-loss checkpoint every 20 it, all native. No W&B and no preview videos.
7. D1 stage: CMA starts at the D0 optimum (native `set_optimal_params` path). Stage B starts from the D0 per-spring
   field mapped by nearest spring midpoint when the topology changed, rescaled by the D1/D0 CMA global stiffness ratio.
   Native CMA returns its best sampled point even if x_init was better.
8. Striker (D3): explicit substep loop over the native kernels plus a penalty kernel with the same law as
   `striker_adapter.py` (V_p = m0/rho_config). The loop matches the native forward graph to < 1 um.
9. Density fixed at 1000 for all worlds (as all baselines). Stage A includes the native acceleration regulariser.
   Native stage B never activates it (`acc_count` stays 0), so the stage-B loss is chamfer + track only.
   No scale normalisation is applied: CMA is rank-based and Adam is scale-invariant, and smooth-L1 stays quadratic.
10. Regulariser check: at x_init the native acceleration term is 0.6% (block D0), 2.4% (stem D0) and 54% (stem D1,
    at the D0 optimum) of the stage-A objective. The observed tracks' frame-to-frame jerk is 40-70x smaller than in
    the upstream example data (J 0.05-0.07 vs 2.9-4.1 m/s), so the native weight penalises our observed motion less
    than it does natively. The stem D1 share is the model's own lattice vibration. Native weight kept.

## Smoke / sanity results (GPU 0, shared with running jobs)
- (a) Injected force read back from the native kernel at full ramp: block D2 [0,-30.000,0] N (17 nodes),
  stem D2c [0,-16.000,-12.000] N (302 nodes), stem D1 [0,0,40] N (402 nodes).
- (b) `sanity-homog-block-D0-Bcentre*.json`: the best homogeneous spring_Y (5.6e3 native = 5.6 N) gives D0 marker
  error mean 0.60 mm, max 1.9 mm (observed max 2.15 mm). Damping/radius variants give 0.58-0.60 mm: the local
  indentation under the press is too soft in a homogeneous lattice. Stem D0 (`sanity-homog-stem-*`): best 1e5,
  mean 1.8 mm, max 9.4 mm on 28 mm observed.
- (c) `block/D*-Bcentre-smoke`, `stem/D*-W1000-smoke`: CMA lowers the loss (block D0 7.5e-7 -> 7.0e-7;
  stem D0 4.9e-4 -> 1.5e-5). Stage B decreases monotonically and the per-spring stiffness spreads
  (block D1 20 it: 9.17e-7 -> 9.08e-7, Y 6760-7576). The native lr 1e-3 moves slowly; production uses 200 it x 24-36
  frames = 4.8k-7.2k Adam steps. D0 -> D1 init path exercised (topology changed, field mapped).
- Substep loop vs native forward graph: 0.0006 mm max difference (stem D2c, 31 mm motion).
- D3 caveat: the sphere's first contact is the surface marker node at x=0.18, 1 cm off its axis, and the 8 mm lattice
  is asymmetric about c_x=0.17. The sphere leaves with vx ~ -0.33 m/s (GIC forecasts: +0.03). This is a
  contact-resolution artefact of the coarse node set. vz rebound 0.40 m/s (GIC 0.28-0.30).

## Cost (measured in smokes on a shared 2080 Ti)
Peak GPU 0.55 GB (block), 1.3 GB (stem); host RAM < 2 GB. Block world ~25-40 min (D0 + D1 ~ 2 x (CMA ~5 min + 200 it
~7 min) + forecasts < 1 min). Stem world ~2-3 h (CMA ~1.3-2.4 s/eval/10 frames, stage B ~3 s/it at 9 frames,
scales ~linearly to 25/37 frames). D3 forecast ~9 s.
