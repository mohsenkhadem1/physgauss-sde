Real-object transfer campaign v19
================================

Requested scope: PokeFlex, PhysTwin, then Deform360; test physical prediction
and input/interaction shifts, retain negative results, keep raw datasets local.

PhysTwin first pass (complete)
-----------------------------
phystwin_run.py exports a strict observed-prefix NPZ, then fits and forecasts
without loading withheld geometry. neural_fem.py ports the frozen shared bulk
decoder to arbitrary P1 tetrahedra. Six Gaussian material features and three
constitutive descriptors are fitted alongside gravity/attachment/floor ratios.
Explicit attachments are quadratic; this is not the structured Q2 solver or a
new end-to-end RGB reconstruction. Geometry, point correspondences and future
tool positions are supplied. The initial observed geometry is treated as rest.

phystwin_campaign.py ran 16 fits/forecasts on GPUs 0-2: neural and matched
analytic FEM, eight sequences, identical 120-node geometry and prefix budget.
The official released PhysTwin forecasts and prefix position regression are
separate controls. The native checkpoint has its authors' representation and
optimization schedule; it was not retrained with our FEM budget.

All eight sequences were exposed in older campaigns. These results are an
exploratory rebenchmark, not a fresh sealed generalization claim. The only
development validation used here was the first 70% of the official sloth-lift
training prefix. One fixed neural recipe was then run on every sequence.

score_phystwin.py reports manual-track mean distance, coordinate RMSE, aligned
shape error and coverage. It separately scores future frames where a mean
prescribed attachment target leaves the training axis-aligned range by >1% of
the initial object extent. This is tool-position input-range extrapolation,
NOT a withheld interaction-family or unseen-object benchmark.

Results: neural/analytic/native PhysTwin manual errors are 42.061/45.716/21.144
mm over all suffixes and 42.103/45.832/21.139 mm on extrapolation frames.
All methods have complete 248-frame suffix coverage; 164 frames meet the
input-range criterion. Neural improves analytic tracking by 8.0% (8.1% on
extrapolation) but does not beat native PhysTwin or position regression.
Aligned shape error changes only 21.261 -> 21.077 mm against analytic FEM.
No rendering-quality, parameter-calibration or active-probing win is claimed.

audit_phystwin.py independently replays manual-track arithmetic and perturbs
hidden object positions/manual tracks to verify identical predictor inputs.
It passes 803 checks. The new force/tangent/parameter derivative tests plus
existing external FEM tests pass 12 tests. The old benchmark outputs and paper
are unchanged.

Reproduction (from repository root)
----------------------------------
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 runs/phystwin-env/bin/python -m pytest scripts/real_transfer_v19/test_neural_fem.py tests/test_external_fem.py -q
runs/phystwin-env/bin/python -m scripts.real_transfer_v19.phystwin_run --case single_lift_sloth --label NEW_LABEL --method neural --device cuda:0 --nodes 120 --frames 18 --iterations 50 --fit-seconds 600 --total-seconds 1800
runs/phystwin-env/bin/python -m scripts.real_transfer_v19.score_phystwin
runs/phystwin-env/bin/python -m scripts.real_transfer_v19.audit_phystwin
.venv/bin/python -m scripts.real_transfer_v19.report_phystwin

Never overwrite the exploratory-01 outputs. Its protocol and hashes live in
reports/real-transfer-v19; raw inputs/predictions/checkpoints live in runs/.
The fixed campaign entry point is historical, not a general scheduler for new
recipes. Use a new label and a new protocol for subsequent experiments.

PokeFlex (licensed local recordings)
-----------------------------------
ETH issues personal download links for noncommercial academic use. Request access
at https://pokeflex.ait.ethz.ch/ and paste your own URL at the hidden prompt.
From the repository root, download all archives (about 500 GB compressed):

    python3 scripts/real_transfer_v19/download_pokeflex.py

For the first foam-dice poking trial only (listed as 4.33G by ETH):

    python3 scripts/real_transfer_v19/download_pokeflex.py poking/FoamDice/FoamDice_T1.zip

Archives keep the portal's directory structure under runs/third_party/PokeFlex-data/.
This Git-ignored directory sits alongside the official reconstruction code at
runs/third_party/PokeFlex-v19/. Re-running skips completed archives and retries
incomplete files. Extract the required trials locally, allowing extra disk space.
ETH prohibits redistribution. Keep recordings and personal URLs out of Git.

Deform360 (acquisition/preprocessing)
------------------------------------
The public snapshot has 192 object directories. Six volumetric objects were
chosen before prediction: 043-dog, 046-sponge, 047-rectangle-sponge,
048-butter-sponge, 049-ball, 051-cube. acquire_deform360.py pins the HF revision
and 5,308 files; acquire_deform360_retry.py adds global rate limiting/backoff.
The large member inventory and raw recordings are local. Downloads verify
file size and the LFS SHA256 when present.

Official release: https://github.com/lhy0807/deform360
Data: https://huggingface.co/datasets/brownu/deform360
Pinned source: d8522a4403b766aeb387510c04e89032a56fdf35
The release provides raw data/annotation tools, not world-model baseline
training code or ready-made predictions. Do not compare our adapted protocol
directly to numbers in the Deform360 paper without matching its setup.

The default SAM3 checkpoint returned HTTP 401 and no HF credentials were
configured. deform360_masks.py uses public Grounding-DINO tiny and SAM2.1
small instead, with first-frame text initialization and forward propagation.
This preprocessing difference must be disclosed. A 430-frame development
sequence (046-sponge, episode 0) was aligned with the official code, its robot
poses extracted, and a first segmentation view visually checked under hand
occlusion. deform360_mask_queue.py processes all 36 calibrated views on GPUs
0-2 with per-camera logs/status and bounded subprocesses.

deform360_reconstruct.py wraps the released Splatfacto reconstruction code
using the existing local CUDA 12.1 compiler toolchain, in a separate environment.
A three-view/frame-zero smoke test uses a separate output directory. It is not
a physics-prediction result. Full reconstruction, tracking, baseline fitting
and OOD evaluation remain required before claiming Deform360 performance.

Acquisition is complete: all 5,308 files, 7,748,388,421 bytes. The first episode's
36 masks are complete; initial-only geometric qualification accepts 29 cameras.
Five of those are reserved from splat training for held-view evaluation. A
three-view smoke test exported 11,749 finite Gaussians. Full per-frame processing
has started with 1,000 first-frame / 200 subsequent iterations. A wrapper closes
Nerfstudio's global Tensorboard writers before scratch cleanup; otherwise the
second frame crashes when a writer accesses the deleted first-frame directory.

The initial held-view reconstruction audit is preliminary: silhouette IoU
ranges from 0.381 to 0.954. This is not yet a qualified geometry reference.
deform360_render_check.py correctly renders exported spherical harmonics;
the depth loader's raw f_dc fields must not be interpreted as RGB colors.
deform360_position_controls.py exports 24 fixed gripper surface landmarks per
gripper from pose/opening alone. Translation equivariance and prefix isolation
pass. These landmarks are not known attachments and do not solve contact.

deform360_prepare_cohort.py runs bounded alignment/masks/geometry checks on
episode zero of all six objects, using GPUs 1-2 while reconstruction uses GPU 0.
Its outputs/status live in runs/real-transfer-v19/cohort-preparation. It does
not launch physical prediction. The frozen evaluation plan reserves five
objects after development on 046-sponge and distinguishes prefix adaptation
from cross-interaction material transfer. Prediction evaluation is unfinished.

Remaining order
---------------
1. Qualify multiview masks, pose units and reconstruction; retain failed views.
2. Produce observation/evaluation bundles with strict temporal separation;
   avoid future-window tracking leakage at the observation cutoff and avoid
   using future tactile response as a position-only action input.
3. Compare neural mechanics, matched analytic mechanics, position regression
   and the native PhysTwin physical model using identical available inputs.
4. Expand across objects/recorded interaction types with a frozen development
   and evaluation split; report both geometry and independent held-view images.
5. Run PokeFlex after approved access; use measured forces as an evaluation
   target, not as privileged input to the position-driven method.
