# pokeflex-v01: PokeFlex real-data protocol (data layer)

Licensed data (non-commercial, no redistribution). Nothing from PokeFlex is stored in git: raw zips are in
`runs/third_party/PokeFlex-data/`, extracted data in `runs/pokeflex-v01/` (git-ignored). Run every script with
`python -I`. Anything heavy runs as a capped unit (`scripts/stem_trial_v01/run_unit.sh`).

| file | purpose |
|---|---|
| `extract.py` | selective, idempotent extraction (meshes, confidences, robot trajectory, camera params; `--images`, `--atlas` optional). Forces are split off from `robot_data.json` into the sealed area. Skips `*.part`/unreadable zips. |
| `segment.py` | poke segmentation from the tool trajectory only -> `segments/<Obj>/<take>.json`; completeness flags from rest meshes -> `<take>.flags.json`; `--validate` = force-peak check on the development take only |
| `pokeflex.py` | loader + sealed outcome store + surface metric; CLI `freeze PRED_DIR`, `verify`, `summary` |
| `audit.py` | geometry/units audit (development take + dropping takes) -> `runs/pokeflex-v01/audit/audit.json` |
| `protocol.json` | roles, scores, freeze rule |
| `objects.json` | PokeFlex object table (mass, size, stiffness N/m) |

Pipeline: `extract.py --kind dropping`, `extract.py --kind poking`, `segment.py`, (`audit.py`).

Roles (per object): passive = dropping takes; calibration = first complete poke of the designated take
(lowest-numbered non-development take with a complete poke) and anything before it; held-out = all other
complete pokes; excluded = incomplete pokes (take starts/ends mid-press). Development take: `FoamDice/FoamDice_T1`
(forces and meshes inspected; excluded from scores and from fitting).

Loader rules: trajectory, camera parameters, segments, dropping meshes and the rest state at each segment's
start frame are free. Outcomes (`mesh`, `forces`, `image`) of frames inside held-out/excluded pokes raise
`SealedError` until `reports/pokeflex-v01/prediction-freeze.json` exists and every hash in it (predictions,
protocol, code, sealed manifests) verifies; each gated file is also checked against its take's sha256 manifest.

Key audit facts: meshes in mm, world +y up, support plane y ~ +0.278 m; mesh topology changes every few frames
(piecewise tracked between keyframes) -> surface-to-surface scoring; forces are in the world frame (N);
robot frame k <-> mesh frame k; the reconstruction does not resolve the dent under the probe, hence the
tool-excluded chamfer as primary surface score.
