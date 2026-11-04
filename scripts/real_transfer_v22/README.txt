PhysTwin corrected-mask rebenchmark and novel-camera visualization (v22)

Purpose
Correct the fitting observation mask, preserve old outputs, rerun the frozen
240-node recipe, and inspect both predictions from unrecorded camera angles.
This is a correction-only experiment, not a newly tuned architecture.

Mask semantics
PhysTwin object_motions_valid[t] qualifies transition t -> t+1. A position at t
now uses visibility at t and incoming validity [t-1]; frame zero uses visibility
alone. Nonfinite positions are excluded. Exported inputs have a new incoming-v22
namespace, so the old cached masks cannot silently be reused. All eight exported
numeric payloads differ from their predecessors only in observed_mask.
This removes adapter-level one-frame look-ahead. PhysTwin's offline tracking
preprocessor itself has not been certified causal.

Protocol
Eight historically exposed sequences, neural and analytic FEM, 240 nodes,
18 prefix fitting frames, 50 optimizer iterations, identical v20 time limits.
The v20 development-selected recipe is frozen; no new tuning on test frames.
Released PhysTwin forecasts remain the native baseline. Five frozen source-to-
target transfers are also rerun with corrected source fits. Three previously
failed initial-geometry gates remain failures; two passing pairs are a small
pilot, not evidence of population-level OOD superiority. Native transfer is an
explicitly adapted baseline, not an official author result.

Novel cameras
For all eight sequences and the two prespecified gate-passing transfer pairs,
render a six-second, 120-frame, 360-degree orbit around the final forecast state.
The object state is frozen: camera movement must not be interpreted as time
prediction. Both methods use identical cameras, appearance, pruning, spherical
harmonics, full initial support geometry and local rigid transport. No future
reference tracks or visibility select the skinning anchors in these videos.
A common camera envelope covers both forecasts, without per-method recentering.
The appearance asset was fitted from the original three cameras. Side/back
views can expose missing textures or holes shared by both methods. No recorded
RGB exists for these novel poses: do not assign PSNR/SSIM/LPIPS or declare a
novel-view winner from these videos. Recorded-camera metrics are kept separate.

Outputs
reports/real-transfer-v22/summary.json: equal-sequence recorded-camera and manual
3D geometry results; transfer-scores.json retains all five transfer diagnostics.
input-audit.json, tests.txt, protocol.json and verification.json record checks.
Local videos and angle contact sheets: runs/real-transfer-v22/novel-views/.
Raw datasets and full rollouts remain local. Old v20/v21 results are preserved.

Reproduce from repository root
runs/deform360-reconstruction-env/bin/python -m pytest scripts/real_transfer_v22/test_masks.py scripts/render_diagnosis_v21/test_transport.py scripts/real_transfer_v20/test_metrics.py -q
runs/phystwin-env/bin/python -m scripts.real_transfer_v22.campaign
runs/phystwin-env/bin/python -m scripts.real_transfer_v22.audit_inputs
runs/phystwin-env/bin/python -m scripts.real_transfer_v22.render_queue
runs/phystwin-env/bin/python -m scripts.real_transfer_v22.transfer_queue
runs/deform360-reconstruction-env/bin/python -m scripts.real_transfer_v22.report
runs/phystwin-env/bin/python -m scripts.real_transfer_v22.final_audit

Campaigns and queues may run concurrently on GPUs 0-2. GPU3 is untouched.
The rendering environment's name is historical: Deform360 data remains paused.
These scripts require previously acquired local PhysTwin assets and the local
ffmpeg binary under runs/media-tooling. Per-job logs expose errors; incomplete
fits are not silently included in aggregate results.

Numerical recovery
The initial corrected zebra neural rollout failed at frame 62 with residual
2.219e-5 after 80 iterations. recover_rollouts.py replays the same fitted twin,
retrying only numerical failures with a 320-iteration cap and unchanged
tolerance. That frame converged in 102 iterations to residual 1.357e-11, with
positive minimum volume ratio 0.714. Original rollout artifacts are retained
in pre-recovery/. No physical parameters were refitted for this recovery.

Projected landmark diagnostic
projected_landmarks.py projects the existing manual 3D annotations into the
same virtual cameras at the frozen final state. It uses shared initial point
correspondences and includes occluded landmarks. This is a visualization of
existing geometric evidence, not independent RGB ground truth. The dinosaur
has zero valid final-frame annotations and is explicitly unavailable; the
standard aggregate uses seven matched sequences, transfer uses two. It is
separate from the full-trajectory 3D metric and may rank methods differently.

Completed results
All 16 corrected fits/rollouts, 10 frozen transfer rollouts and 10 orbit videos
completed. Fourteen unit tests and 31 final integrity/video checks pass.
Standard mean manual tracking: neural 39.013 mm, analytic 40.538 mm, released
PhysTwin 21.144 mm. Previous neural was 38.822 mm: the correction is not a
performance gain. Recorded-camera PSNR: neural 16.129 dB vs PhysTwin 17.659 dB.

On the two registration-gate-passing transfers: neural 39.157 mm vs adapted
PhysTwin 44.627 mm (12.26% lower mean), with one win and one loss. Lift-to-push:
42.779 vs 58.230 mm; one-hand to two-hand zebra lift: 35.536 vs 31.024 mm.
Final-state projected landmark errors in the orbit cameras tell the same mixed
story: lift-to-push 46.41 vs 67.32 pixels (31.06% lower), zebra transfer 67.17 vs
45.08 pixels (ours worse). Their mean is essentially tied, 56.79 vs 56.20 pixels.
Across seven standard sequences with final annotations, ours is worse (41.39 vs
21.83 pixels). Thus new views do not establish an overall advantage. Both twins
show appearance holes in poorly observed side/back regions.
