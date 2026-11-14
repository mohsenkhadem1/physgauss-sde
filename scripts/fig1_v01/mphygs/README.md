# M-PhyGs material-estimation adapter (Figure-1 S1/S2/S3)

Runs only M-PhyGs' physical material-estimation stage (vendored code in
`runs/gap-crossmethod-v02/vendor/m-phygs`, unmodified) on our synthetic scenes, then forecasts held-out commands.
Outputs: `runs/fig1-v01/mphygs/`.

| file | role |
|---|---|
| `common.py` | scene constants, particle sampling, k-means clusters, force selection, marker -> dense track interpolation |
| `synth.py SCENE EP WORLD` | writes native-layout inputs (`physical_particle_pyramid/`, `physical_particle_tracking/`, `physical_particle_contact_motion/`) + `config.json` (native `push_carnation.json` template). D0/D1 only. |
| `mphygs_lib.py` | the patches (see below): ForcedSimulator, density-fixed material_func, stubs |
| `fit.py SCENE EP WORLD --frames N [--init CKPT]` | one native `train_physical_params` call = one `alternating_photo_opt --skip_photometric --opt_settings mat --num_outer_iters 1 --uniform_gravity` stage, single chunk |
| `forecast.py SCENE WORLD FIT_EP --ckpt CKPT --targets ...` | rollout with the same simulator; `--truth-check` (sanity b), `--consistency` (substep loop vs native forward) |
| `pipeline.sh SCENE WORLD` | synth -> D0 curriculum -> D1 refit from final D0 ckpt -> forecasts |
| `profile_step.py` | timing of one simulated frame |

Run (one capped unit per world; any GPU, peak < 0.4 GB):

    scripts/stem_trial_v01/run_unit.sh mphygs-block-Bleft 12G $PWD $PWD/runs/fig1-v01/mphygs/logs/pipe-block-Bleft.log \
        CUDA_VISIBLE_DEVICES=0 -- scripts/fig1_v01/mphygs/pipeline.sh block Bleft

Outputs: `forecast-{EP}-{D0|D1}-{WORLD}.npz` with `xyz (F,N_body,3)` (frame 0 = rest, 48 fps), `rest`, `labels`;
stem adds `station_centroid (F,4,3)` (particles binned by rest x into `STATIONS`), D3 adds `striker (n_substeps,7)`
= (t, c, v). Matching `.json` has per-cluster E / density / nu, force audit, substep info. Fit stages write
`params.json` + native `material_field.ckpt`, `losses_transition.json`, `material_params.ply`, `particle_states/`.

## Deviations from native M-PhyGs (honesty notes)
1. Only the material stage. No 3DGS reconstruction, DINO/GARField clustering, dynamic-GS tracking, contact
   estimation, rest-shape stability optimisation, or photometric stage.
2. Particles: uniform grid on the known rest geometry, spacing 6.7 mm (block 9720, stem 9504 body particles),
   volume h^3, box SDF. MPM grid 75 cells/m (dx 13.3 mm, native 90).
3. Clusters: k-means (K=12, seed 0) on rest positions, geometry only (native: GARField + DINO). The
   `dino_features` used by the native mat_reg neighbour graph are rest coordinates (no appearance features exist).
4. Supervision: native pseudo-GT 3D tracks replaced by thin-plate-RBF interpolation (rest coordinates) of the
   RGB-triangulated markers (block 20, stem 16) plus zero-displacement anchors on the known fixed boundary.
5. Boundary: static identity-motion contact (native modifier) on the pinned set: block = 2 particle layers under
   the glued face (1296 extra particles, not in outputs); stem = body particles with x <= -0.20. No colliders.
6. Known applied force via the native `particle_ext_force` argument of `PhysicsSimulator.forward` (not used by any
   native caller), mass-weighted over the same particle selection as our GIC/PAC adapter, ramp evaluated at the
   midpoint of each checkpoint segment (1/10 frame). Verified: per-substep ramp differs by 7 um on 3.8 mm.
7. Gravity: native `--uniform_gravity` (no residual-gravity prestress), start from undeformed rest, v = 0,
   init acceleration 0 (native chunk-0 behaviour; the per-particle polyfit is skipped since it is unused).
8. Density fixed 1000 kg/m^3 (as GIC/PAC); E = (E/rho)_field x 1000; nu native fixed 0.3 (block truth nu 0.4).
9. `max_stiffness` 2e4 (native 1e5): bound on E/rho and CFL substep; >= 5x above all reference E/rho.
10. One GPU, one chunk (chunk-consistency terms inactive, as natively for one chunk); frame curriculum
    12/24 (stem D1 12/24/36) instead of 15/30/45/50/50; 150 Adam iterations per stage, lr 0.05 (native).
11. Photometric inputs are dummies (one 32x32 view, constant images); `rasterize_gaussians` is stubbed (the env's
    gsplat backend cannot import: missing `rich`) and `mphygs.dino_utils` is stubbed (transformers/nerfstudio
    missing). Photo/feature/DT weights are 0 in the tracking stage; logged `photo_grad_dp` = 0 exactly.
12. `wp.ScopedTimer` made a no-op (profiling-only syncs; no arithmetic change, ~3.5x faster on shared GPUs).
13. Striker (D3) forecast uses an explicit substep loop over the native `p2g2p` kernels so the penalty force
    (same law as `striker_adapter.py`) is updated every substep; loop verified against native forward (6 um).
14. Native loss weights kept (pos 1, mat_reg 1e-2, var terms 1e-6). With mm-scale block motion the pos loss is
    ~1e-6 m^2, so mat_reg (spatial smoothness of log E) dominates early iterations.

## Smoke / sanity results (GPU 0, shared with a running fig1 job)
- (a) force audit: block 72 particles loaded, total [0,-30,0] N; stem D1 576 particles [0,0,40]; D2c [0,-16,-12];
  injected sum at full ramp equals the command (`forecast-*.json: injected_total_force_at_end`).
- (b) `sanity-truthcheck-D0-Bcentre.json`: true block E (6e4 / 4.8e5 inclusion, nu 0.3) + D0 force vs D0 RGB
  markers: max 0.98 mm, mean 0.31 mm error (observed displacement max 2.15 mm, mean 0.91 mm).
- `sanity-consistency-block.json`: substep loop vs native forward, 8 frames: 5.7 um (const force) / 7.3 um (ramp).
- (c) fits (`*_smoke` stage dirs): stem D1 W1000, 6 frames x 12 it: pos loss 3.8e-4 -> 1.7e-5, cluster E 3.8-5.7 MPa
  (ref 4 MPa). Block D0 Bcentre, 8 frames x 12 it: total loss 1.7e-5 -> 2.9e-6 but pos loss flat (1.742e-6 ->
  1.741e-6), E/rho stays ~1e4 (random init ~max/2): regularisers dominate at mm-scale motion; a forward scan of
  homogeneous E/rho (scratch) shows the pos loss minimum near E/rho ~100, shallow from the stiff side.
- Forecast smokes in `smoke/` (not results). Pipeline dry run (2 frames, 1 it): `smoke/pipeline-test/`.

## Cost
~6-8 s per simulated frame per Adam iteration (733 substeps/frame, overhead-bound), peak GPU < 1.5 GB
(0.53 GB at 8 frames), host RAM ~1.5 GB. Block world: D0 (12,24) + D1 (12,24) = 72 frame-stages x 150 it
~ 18-24 h. Stem world: D0 (12,24) + D1 (12,24,36) = 108 x 150 ~ 27-36 h. Forecasts: ~1 min (force), ~3-5 min (D3).
