# Independent FEM → Isaac render-only adapter — 21 October 2026

**The bounded TRAIN smoke passed: four calibrated RGB/depth/mask images from two exact saved times and two views.** The new adapter is `experiments/render_independent_trajectory.py`. Physics comes entirely from the completed independent float64 FEM trajectory; Isaac renders a plain USD surface mesh. It creates no physics scene, deformable actor, rigid body, fixture or force. No physics core/native capture source was modified, no bulk dataset was captured and no model was trained.

## Input, clock and public-view rules

Source: `runs/gaussian-campaign-v01/independent-dynamic-gate-v01/expanded-matrix-v01/t0000-temporal-diagonal-hz480` (TRAIN point-pin, diagonal pulse). Trajectory SHA256: `35ce33482af2a7b2fa8bedf94b37d903ddaaaa8e0745072ac1c901424ec436aa`. Its31 saved states are exactly0–3seconds at10Hz. This smoke used source indices0 and15, meaning0 and1.5seconds; it generated no intermediate states or extra61-frame sequence.

The loader requires terminal/completed input, checks the exact trajectory/static-reference hashes, all saved source/input snapshots against their recorded hashes, canonical reference arrays and the saved clock. It verifies the **frozen provenance snapshots**, allowing a valid older runner snapshot even when the live campaign runner changes. Only untranslated canonical-coordinate completed TRAINt0000/t0001 are currently accepted. Source faces/rest vertices are exact; USD point updates are their saved positions rounded to float32 (maximum vertex roundoff14.85nm in this smoke).

The primary camera orbits the public rest-bounds center, with radius2.6times the rest diagonal, a declared110degree azimuth arc and fixed elevation schedule. The candidate follows the same time schedule with a fixed60degree azimuth /10degree elevation offset. Other fixed offsets are supported by the interface but were not rendered in this smoke. Neither camera follows target motion, COM or a loaded future state. Vertex appearance is a fixed function of public rest coordinates; it does not encode material, support or force labels. There are no fixture markers or force arrows in RGB.

Source states are rendered at a paused Kit timeline of zero, with Replicator's synchronous `delta_time=0` capture. Stored `times_s` are the saved FEM physical times. These are same-time held-view targets, **not predictions, forecasts or interpolated motion**. Candidate pixels stay under `private/held_views`; public query files expose only their requested calibration/poses/times.

## Actual four-image result

Successful capture root:

`runs/gaussian-campaign-v01/independent-render-smoke-v01/t0000-v03`

| Source time | View | Object pixels | Maximum independent depth error |
|---|---|---:|---:|
| 0seconds | primary | 5156 | 0.195µm |
| 0seconds | candidate00 | 6215 | 0.119µm |
| 1.5seconds | primary | 6007 | 0.145µm |
| 1.5seconds | candidate00 | 6313 | 0.178µm |

Each image passed64 current-surface ray/depth checks with incidence cosine≥0.2 and the unchanged2µm tolerance. All four independent coarse silhouette checks (2120rays each, including background) have IoU1.0; semantic object masks and finite metric depth also have IoU1.0. Projected vertex bounds agree with raster bounds within0.948pixel. The numerical depth unproject/reproject round trip is2.84e-14pixel; this is a numerical consistency check, not a real-camera calibration accuracy claim. No mask touches an image edge. RGB foreground/background contrast is nonzero. The parent visually checked the1.5second primary image: object/background separation is clear and no hidden markers are visible. This tiny-force point-pin case mostly rotates; it is not a full flexible-material demonstration. These checks validate rendering/sensor alignment, not independent FEM physical accuracy or learned prediction.

Artifacts relative to the successful root:

- `public/reference.npz`: exact rest vertices/faces and fixed rest-derived colors.
- `public/observations/{rgb,depth,masks}/{00000,00015}.*`: primary inputs; depth stores image-plane metres, invalid pixels zero.
- `public/observations/cameras.npz`: primary K, USD row-vector poses, resolution, exact physical times and source indices.
- `public/query_cameras.npz`: same-time candidate calibration only; candidate target pixels are in `private/held_views/candidate00/{rgb,depth,masks}`.
- `private/source-state-subset.npz`: generator/evaluator-only saved positions for the two states.
- `protocol.json`, `captures.json`, `summary.json`, `execution.json`, `file-manifest.json` and `provenance/`: frozen source, state hashes, renderer settings, initial USD mesh scene, GPU metadata and quantitative checks.

## Device and completion failures were preserved

The validated Isaac convention selects physical `active_gpu=2`, `physics_gpu=2`, `multi_gpu=False`, with CUDA_VISIBLE_DEVICES unset. Only GPU2 has a graphics/RTX context. SDK Replicator depends on Warp and initializes auxiliary C-only contexts on other cards even without actors; observed footprints stayed132MiB on0 and100MiB on1/3. The successful protocol allowed these evidenced startup contexts up to160MiB each and sampled **our own PID** before/after rendering. Some utilization values are unsupported/missing and explicitly remain unknown; they are not reported as zero. All graphics remained on2, and no sustained off2 compute was observed in available samples. The SDK does not provide strict process-wide isolation from initialization contexts on0/3 under this convention.

The initial `t0000-v01` guard incorrectly required every context to be solely2; it stopped before any capture. `t0000-v02` stopped at the predeclared64MiB auxiliary limit. Those frozen sources, error reports and raw logs remain intact. Their changes affected device initialization policy only; sensor tolerances, cameras, source states and public/private split were not relaxed.

Another observed SDK behavior matters: `fast_shutdown` can exit0 from `app.close()` after a recorded worker failure. The v01 exit-only driver flag is explicitly superseded by `smoke-verdict-v01.json`. Corrected drivers require worker terminal/completed flags, all four images, calibration/summary and matching manifest hashes. The adapter writes its evidence before closing. The v03 completion contract passed. PID2088534 is gone, all its GPU contexts are gone, and GPU2 returned to8MiB/0%; no jobs are queued.

## Tests, scope and reproduction

Eight CPU tests pass: analytic plane depth, camera conventions/rest translation, exact saved-frame selection, completed-input/source/trajectory tamper guards, semantic label isolation, positive sensor capture, stale depth/wrong pose/intrinsics rejection, and graphics/auxiliary GPU policy. Tests are in this directory. All14 source input hashes and all successful capture hashes were independently reverified after rendering; public static arrays and camera archives exactly match the source/records.

Frozen successful adapter SHA256:

`da1a2d51d1b7ba7bcb335dafabbec6929f4228c41c749a3fe5297f542e8217d2`

The exact successful command/environment is in `smoke-execution-v03.json`; `run_smoke_v03.py` refuses to overwrite its output and enforces the four-image contract. The Isaac interpreter is `/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python` (Isaac Sim6.2). A future run needs a new output/log directory and a separately coordinated free GPU2; no bulk run is queued or authorized by this smoke.

Only this case, two times, one candidate offset and320×240 have runtime validation. Full trajectories, additional views/cases, observation scheduling, PIDG dataset adaptation and sealed evaluation remain future work after their respective gates. Existing native failures, sealed tests and physics reference qualification are unchanged. `publication-inventory.json` scopes new source/metadata and the small successful capture plus preserved device-failure evidence; it includes no modified physics implementation.
