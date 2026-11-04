VIDEO-TO-SCENE QUERY PROTOTYPE v24

This is a trained real-video reconstruction prototype, not an editable physical
scene model or a D4RT reproduction. Raw data/checkpoints/videos remain local.

INPUT: PhysTwin real RGB from cameras 0/1 at source time 0 and target time t,
known calibration, annotation-selected source pixels, and query time.
OUTPUT: source depth and a 3D displacement for each query, hence a world-space
3D point at time t. Target-time images are inputs: this is not forecasting.

Frozen pretrained ResNet18 features feed a two-layer shared transformer memory;
two cross-attention query blocks predict geometry. 635,396 trainable parameters.
No mesh or source 3D coordinates enter the network. Query selection uses labels,
so automatic foreground discovery is not demonstrated. Camera poses are known.

12 training sequences, 4 validation sequences, 6 held-out rope/package sequences.
The fixed protocol predates training; final step 2500 is used without test-based
checkpoint selection. Two matched runs use visual features or zeroed features.
GPU training was run on this machine. Official train prefixes only contribute
training labels. Known-case suffix evaluation remains image-conditioned.

RESULTS (equal frame, then equal case means)
                           visual    image-blind   true-initial persistence
Train-object suffix mm     55.13     77.28         85.51
Validation-object mm       49.63     51.66         33.04
Test-object mm             75.37     80.34         54.29
Test displacement mm       69.98     59.12         54.29
Test held-camera pixels    35.16     39.10         24.29

Images improve test position error by 6.19%, but displacement is worse and a
privileged true-initial-geometry baseline is stronger. This is not evidence
of superiority over native PhysTwin or PIDG. One seed; no significance claim.
Track labels are author reconstructed estimates, not independent metrology.
Camera2 RGB is withheld from model inputs, but upstream reconstructed labels
may use it. Object grouping uses author case names, not verified identities.

Videos show projected 3D outputs over recorded camera2 footage. Backgrounds are
NOT generated RGB. The three cases were selected before evaluating results:
single_lift_sloth, single_push_rope_4, weird_package. No learned mechanics,
action-conditioned simulation, camera inference or uncertainty calibration is
implemented by this prototype. Previous simulation experiments are separate.

REPRODUCE from repository root with a CUDA PyTorch/torchvision environment:
python -m scripts.video_scene_v24.protocol
python -m scripts.video_scene_v24.prepare
CUDA_VISIBLE_DEVICES=0 python -m scripts.video_scene_v24.train --method visual
CUDA_VISIBLE_DEVICES=1 python -m scripts.video_scene_v24.train --method image_blind
CUDA_VISIBLE_DEVICES=2 python -m scripts.video_scene_v24.evaluate
python -m scripts.video_scene_v24.video
python -m scripts.video_scene_v24.report
python -m scripts.video_scene_v24.audit
Training refuses to overwrite existing runs. See reports/video-scene-v24 for
protocol, results, training provenance, contract tests, and same-agent audit.
