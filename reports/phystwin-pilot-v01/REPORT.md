# PhysTwin first implementation and pilot — 26 October 2026

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
| Official checkpoint: lift | 4.90 | 24.93 | 17.86 |
| Official checkpoint: stretch | 9.60 | 9.75 | 5.98 |
| Local stage two: 200 iterations, author CMA | 4.96 | 25.18 | 17.90 |
| Local CMA: 20 generations, no stage two | 8.01 | 31.15 | 24.35 |
| Local CMA + local stage two: 20 iterations | 6.99 | 29.57 | 24.21 |

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
| Official checkpoint: lift | 24.93 | 46.00 | 29.74 |
| Official checkpoint: stretch | 9.75 | 25.80 | 37.50 |
| Local stage two: 200 iterations, author CMA | 25.18 | 46.14 | 29.62 |
| Local CMA + local stage two: 20 iterations | 29.57 | 49.45 | 35.83 |

On the official lifting checkpoint, constant velocity has lower surface L1 error
(14.02 mm) than PhysTwin (17.86 mm), despite worse manual tracking. The stretching
checkpoint wins both metrics against these temporal baselines. There is no basis
for declaring PhysTwin uniformly superior from this pilot.

## Reproduction and numerical checks

- Lift: local checkpoint rollout versus author-saved trajectory RMS = 0.00636 mm.
- Stretch: the same reproduction difference = 0.02487 mm.
- After best-fit rigid-motion removal, sparse manual-track nonrigid RMS is 24.79 mm for lifting and 30.86 mm for stretching (38.8% and 61.0% of total motion RMS). These are not dense strain measurements.

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
