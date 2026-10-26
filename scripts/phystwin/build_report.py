"""Publish small summaries; keep models, datasets and videos local."""
import hashlib,json,subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[2]
runs=root/'runs/phystwin-pilot-v01'
report=root/'reports/phystwin-pilot-v01'
report.mkdir(parents=True,exist_ok=True)
names=['reference','double-reference','retrained','fresh-cma','fresh-two-stage-20']
summaries={name:json.loads((runs/name/'summary.json').read_text()) for name in names}
cma=json.loads((runs/'cma/summary.json').read_text())
subprocess.run([str(root/'runs/phystwin-env/bin/python'),str(root/'scripts/phystwin/audit_results.py'),
    '--output',str(runs/'audit.json')],cwd=root,stdout=subprocess.DEVNULL,check=True)
audit=json.loads((runs/'audit.json').read_text())
gradient=json.loads((runs/'gradient-check.json').read_text())
(report/'results.json').write_text(json.dumps(dict(runs=summaries,cma=cma,audit=audit,gradient=gradient),indent=2)+'\n')
labels={'reference':'Official checkpoint: lift','double-reference':'Official checkpoint: stretch',
    'retrained':'Local stage two: 200 iterations, author CMA',
    'fresh-cma':'Local CMA: 20 generations, no stage two',
    'fresh-two-stage-20':'Local CMA + local stage two: 20 iterations'}
rows=[]
for name in names:
    s=summaries[name]
    rows.append(f"| {labels[name]} | {s['result']['train']['manual_track_mean_m']*1000:.2f} | {s['result']['test']['manual_track_mean_m']*1000:.2f} | {s['result']['test']['chamfer_l1_mean_m']*1000:.2f} |")
baselines=[]
for name in ['reference','double-reference','retrained','fresh-two-stage-20']:
    d=audit[name]
    baselines.append(f"| {labels[name]} | {d['physical_model']['manual_track_mean_m']*1000:.2f} | {d['last_prefix_model_state_persistence']['manual_track_mean_m']*1000:.2f} | {d['last_prefix_model_velocity_extrapolation']['manual_track_mean_m']*1000:.2f} |")
text='''# PhysTwin first implementation and pilot — 26 October 2026

The official PhysTwin physics pipeline runs on this machine. Two author checkpoints
were reproduced, the full 200-iteration second-stage lifting fit was rerun from
the authors' CMA initialization, and both fitting stages were exercised locally
from default first-stage settings (20 CMA generations + 20 dense-fit iterations).
This is a development pilot on two motions of one soft toy, not a new benchmark.

## Scope and implementation

Upstream: https://github.com/Jianghanxiao/PhysTwin, MIT,
commit `81c718790a37e5e0102eb77af2c6edd34a9db25f`.
The only checkout patch guards optional interactive and Gaussian-renderer imports.
The adapter uses the original topology, Warp kernels, physical time step, losses,
optimizer and training-based checkpoint selection. CPU videos replace the OpenGL
preview. No W&B data is sent. GPUs 0–2 were used, with CPU thread limits; GPU 3
was left for the desktop. Data and weights remain local.

Inputs are the authors' processed point clouds, tracks, geometry completion and
control trajectories. Raw RGB-D reconstruction and Gaussian appearance training
were not rerun. Future controller trajectories are supplied. Offline tracking
preprocessing has not been audited for strictly causal access. No uncertainty,
unknown-force inference, boundary inference, active probing or our block data is
part of this pilot.

## Results

All table errors are millimetres. Tracking is mean Euclidean distance of manual
landmarks; surface error is the official one-direction L1 Chamfer convention.
The columns are different metrics and should not be compared to each other.

| Run | Training tracking | Future tracking | Future surface L1 |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Lift uses frames 0–58 for training and 59–84 for future evaluation. Stretch uses
0–133 and 134–191. All trajectories start at frame zero and roll forward without
resetting object state to future observations. Parameters/checkpoints are selected
using training losses only. Test metrics were inspected as development diagnostics;
these are not sealed final-test results.

The full stage-two run uses the authors' first-stage initialization. The fresh
local two-stage run uses only 20 stage-two iterations and should not be presented
as a completed 200-iteration, from-scratch reproduction.

## Stronger temporal sanity checks

Held-out tracking error, millimetres. Persistence and constant velocity start from
the final fitted prefix states of the corresponding physical model, without future
object observations. They are conditional diagnostics, not separately trained SOTA
models. They also do not receive the future controller trajectories that PhysTwin
receives, so this is not an equal-action-input competitive benchmark.

| Run | PhysTwin | Last-prefix persistence | Constant velocity |
|---|---:|---:|---:|
'''+ '\n'.join(baselines)+'''

On the official lifting checkpoint, constant velocity has lower surface L1 error
(14.02 mm) than PhysTwin (17.86 mm), despite worse manual tracking. The stretching
checkpoint wins both metrics against these temporal baselines. There is no basis
for declaring PhysTwin uniformly superior from this pilot.

## Reproduction and numerical checks

'''+f'''- Lift: local checkpoint rollout versus author-saved trajectory RMS = {summaries['reference']['reference_trajectory_rmse_m']*1000:.5f} mm.
- Stretch: the same reproduction difference = {summaries['double-reference']['reference_trajectory_rmse_m']*1000:.5f} mm.
- After best-fit rigid-motion removal, sparse manual-track nonrigid RMS is {audit['reference']['manual_nonrigid_rms_m']*1000:.2f} mm for lifting and {audit['double-reference']['manual_nonrigid_rms_m']*1000:.2f} mm for stretching ({audit['reference']['nonrigid_to_total_rms_ratio']*100:.1f}% and {audit['double-reference']['nonrigid_to_total_rms_ratio']*100:.1f}% of total motion RMS). These are not dense strain measurements.
'''+'''
A one-direction finite-difference audit checks uniform log-stiffness perturbations
at training frame 40, with fixed starting state. Matched captured-graph derivatives
agree within approximately 0.3–1.5% for the three tested perturbation sizes. This
does not validate every parameter or long-horizon sensitivity.

An initial 17% discrepancy came from comparing different loss normalizations.
Upstream CUDA graph capture freezes the initial visibility-count scalar normalizers;
eager loss evaluation uses the updated per-frame counts. At the checked frame,
visible counts change from 5,046 to 4,244 and valid-motion counts from 4,991 to 4,077.
The corrected audit compares the same captured objective. The original normalization
behavior remains unchanged to preserve the baseline. The initial mismatched audit
is retained locally as `gradient-check-eager-mismatch.json`.

Three tests check split boundaries, L1 versus Euclidean conventions, missing manual
annotations, and the rigid-motion diagnostic. See `tests.txt` and `gradient-check.json`.

## Reproduce and inspect

See [runner instructions](../../scripts/phystwin/README.md). Exact environment and
input hashes are stored beside this report; `results.json` contains every result.
All videos use observed surface points versus predicted surface points, fixed
shared projection/scale, and labelled training/future intervals. They do not show
photorealistic Gaussian rendering.

Local videos (not uploaded):

- `runs/phystwin-pilot-v01/reference/inference.mp4`
- `runs/phystwin-pilot-v01/double-reference/inference.mp4`
- `runs/phystwin-pilot-v01/retrained/inference.mp4`
- `runs/phystwin-pilot-v01/fresh-two-stage-20/inference.mp4`

Next: freeze a matched input/action protocol for our simulated objects, then
compare transferred interaction prediction. Do not interpret effective spring
stiffness as measured continuum material constants or calibrated uncertainty.
'''
(report/'REPORT.md').write_text(text)
files=sorted((root/'scripts/phystwin').glob('*'))+[root/'tests/test_phystwin_metrics.py']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(report/'source-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(report/'REPORT.md')
