PhysTwin diagnosis, rendering and frozen interaction transfer (v20)
=================================================================

Completed work
--------------
1. Twelve development fits: six one-factor recipes on two lifting prefixes.
   Fit first 70% of the official training prefix; validate on the remainder.
   The official suffix was excluded from this selection. A 240-node mesh won:
   19.726 mm versus 22.111 mm for 120 nodes. More optimizer time did not help.
   Narrow/wide attachments did not improve the mean; disabling the floor hurt.
2. Sixteen full-prefix fits/forecasts with the frozen 240-node recipe: neural
   and matched analytic FEM over eight sequences. Neural tracking improved
   42.061 -> 38.822 mm (7.7%) relative to our old adapter; analytic is 40.247 mm.
   Released PhysTwin remains substantially better at 21.144 mm. All 248 suffix
   frames are available. These eight sequences were historically exposed.
3. Four frozen-parameter dynamic diagnostics: implicit Euler at two/four
   substeps on two development prefixes. Adding inertia without refitting
   worsened both cases. The timestep check is sensitivity, not convergence proof.
4. Five recorded cross-interaction pairs, old and refined neural/analytic
   material fits, and an adapted frozen native PhysTwin baseline. Object
   parameters are frozen from single-hand lifting. Target inputs contain only
   initial geometry and prescribed hand trajectories; no target motion fitting.
5. Rendering: shared released Gaussian appearance, local rigid skinning and
   gsplat rasterization for all methods. Frame zero plus three predetermined
   test frames, all three cameras, on eight cases. Also rendered the two
   registration-screened transfer pairs. This is not novel-camera testing:
   the three cameras were used to fit the original appearance asset.

Metrics and result interpretation
---------------------------------
metrics.py reports manual 3D distance, displacement error, pair-distance-change
error, prediction coverage, PSNR/SSIM/LPIPS and tetrahedral validity. Pair-distance
change is a conventional deformation diagnostic, not a claimed new metric.
Images use reference-mask crops, a common black background, and identical hand
occlusion removal. LPIPS uses pretrained AlexNet. Predicted spill is retained
inside the crop; full-image silhouette IoU is reported separately.

Standard future image metrics (equal sequence means; three sampled frames):
                       PSNR dB       SSIM       LPIPS
Neural 240             16.127        0.599      0.327
Analytic 240           16.060        0.598      0.328
Released PhysTwin      17.659        0.647      0.264
We do not claim overall rendering parity or superiority.

The source twin is rigidly registered to target frame-zero geometry. Three of
five pairs fail the fixed initial-geometry gate and remain visible diagnostics.
The two passing pairs still have initial shape mismatch; the gate does not
certify exact semantic correspondence or a clean initial-state match.

On those two pairs, refined frozen neural tracking is 39.142 mm versus 44.627 mm
for adapted frozen PhysTwin (12.3% lower mean). One pair improves, one loses:
  sloth lift -> push: 42.773 vs 58.234 mm (26.5% lower)
  zebra one-hand -> two-hand lift: 35.511 vs 31.020 mm (ours worse)
For pushing, future-image PSNR is almost equal (14.066 vs 14.089 dB), while
pair-distance-change error is 23.673 vs 25.543 mm. This is an illustrative
appearance/physical-response comparison, not population-level superiority.

The adapted native baseline retains every source object spring and learned
stiffness. New controller springs use geometric connections and the median
source controller stiffness. This is a disclosed transfer policy, not an
official author result. Self-replays match the released outputs to maximum
coordinate discrepancies of 0.257 mm (sloth) and 0.0018 mm (zebra).
The native source optimizer/representation budget differs from our adapter's.

Numerical checks show finite rollouts and no inverted FEM elements. However,
our maximum surface penetration of the dataset z=0 plane reaches 8.927 mm,
versus effectively zero for native PhysTwin. One 240-node mesh still contains
an unused node. These are outstanding physical-model/discretization issues.
No measured-force accuracy, calibrated uncertainty or active-probing advantage
has been established. Do not interpret the new metric suite as proving these.

Validation and reproduction
---------------------------
All commands run from repository root. Existing v19 outputs are preserved.
Mechanical environment: runs/phystwin-env/bin/python (NumPy/SciPy/Torch/Warp).
Rendering/test environment: runs/deform360-reconstruction-env/bin/python
(Torch 2.4, gsplat 1.4, nerfstudio 1.1.5, scikit-image, torchmetrics, pytest).
The environment name is historical: Deform360 dataset work remains paused.

runs/deform360-reconstruction-env/bin/python -m pytest scripts/real_transfer_v20/test_metrics.py -q
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.development_campaign
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.refined_campaign
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.registration
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.native_campaign
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.transfer_campaign --label refined-frozen --fit-root runs/real-transfer-v19/fits/v20-selected-mesh240
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.score_transfer --label refined-frozen
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.score_refined
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.render_campaign
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.build_report
runs/phystwin-env/bin/python -m scripts.real_transfer_v20.final_audit

Six metric/input-isolation tests pass. 159 final audit checks pass, including
sealed fit/prediction hashes, frozen material parameters, native zero-observation
input bundles, source self-replay tolerances and independent aggregate arithmetic.
The common Gaussian warp passes identity and rigid-motion tests to roundoff.

Campaigns have finite subprocess timeouts, logs and per-job results. Existing
completed outputs are retained; failed output folders require inspection before
retry. Source-only self-replay is run separately before the native campaign.
Raw image/appearance data and full prediction arrays stay under ignored runs/.
Reports, protocols and reusable scripts are the publishable evidence. Existing
paper claims are unchanged; use this report's limitations if integrating results.
