NATIVE RECONPHYS DROP: REPRODUCTION AND PUBLICATION GUIDE
=======================================================
Run commands from the physgauss-sde repository root. This is a controlled native
forward-model ambiguity experiment, plus a separately frozen pretrained-model
extension. It is not a new trained video-identification method.

1. UPSTREAM SOURCES AND ATTRIBUTION
Official code: https://github.com/chuanshuogushi/ReconPhys
Commit: 184b53f93d6f75d445e3197af52eae48c2492ad6
Paper: https://arxiv.org/abs/2604.07882
Checkpoint: https://huggingface.co/chuanshuogushi/ReconPhys
Dataset: https://huggingface.co/datasets/chuanshuogushi/ReconPhys_dataset
InternViT code/config: OpenGVLab/InternViT-300M-448px-V2_5 on Hugging Face.
Cite and credit ReconPhys and the upstream Gaussian assets when showing renders.
On 2026-11-07 the checkpoint model card declares CC-BY-NC-4.0; the dataset API
has no card/license metadata, and the pinned code checkout has no LICENSE file.
Do not infer a blanket dataset/code redistribution license from the model card.
Download upstream assets directly; this release does not redistribute them.
Checkpoint SHA256:
8264b9eb7c114f0ea4f2a4d0a0096601eebdc72642b7d2a1e1b3cd76e0fbbe6f
All 22 downloaded files, URLs, sizes and hashes: downloads.json.
The first three lexicographic asset folders, sample_00000, were fixed before
outcomes. No asset selection by performance. Released train/test membership is
not established here.

2. ENVIRONMENT (NO HISTORICAL ENVIRONMENT REQUIRED)
The executed environment used Python 3.10.22, torch 2.5.0+cu121,
torchvision 0.20.0+cu121, CUDA 12.1, RTX 2080 Ti (compute capability 7.5).
Exact core versions and source commits: environment.json. The original local
venv inherited some packages from another environment through a .pth file;
that machine-specific shortcut is NOT a portable dependency or release input.
The historical drop_build.py has machine-specific compiler paths; do not run it
unchanged elsewhere. The following clean bootstrap recipe replaces those paths.
It has not been independently rerun in a clean container; import/build checks
must pass before producing new results.

  python3.10 -m venv runs/gap-first-v01/drop/env
  source runs/gap-first-v01/drop/env/bin/activate
  python -m pip install --upgrade pip setuptools wheel ninja
  python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
  python -m pip install numpy==1.26.4 scipy==1.15.3 transformers==4.48.3 tokenizers==0.21.4 yacs imageio imageio-ffmpeg opencv-python-headless scikit-image lpips matplotlib Pillow==12.3.0 plyfile easydict tqdm einops timm safetensors huggingface-hub rembg onnxruntime open3d==0.18.0 pygame
  python -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu121.html

Install a complete CUDA 12.1 toolkit (including nvcc and development headers)
and a compatible host GCC/G++. Set CUDA_HOME to that toolkit, and CC/CXX to
your compiler binaries. Do not set either to the old PIDG environment.
Set TORCH_CUDA_ARCH_LIST=7.5 for RTX2080Ti (adapt only for a different device),
FORCE_CUDA=1 and MAX_JOBS=2. Build these sources without build isolation:
  pytorch3d: https://github.com/facebookresearch/pytorch3d
    e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31
  mip-splatting: https://github.com/autonomousvision/mip-splatting
    dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
    Build submodules/diff-gaussian-rasterization (initialize submodules).
  nvdiffrast: https://github.com/NVlabs/nvdiffrast
    253ac4fcea7de5f396371124af597e6cc957bfae
  utils3d: https://github.com/EasternJournalist/utils3d
    9a4eb15e4021b67b12c460c7057d642626897ec8
For each checked-out build directory:
  python -m pip install --no-build-isolation --no-deps <directory>
The mip-splatting rasterizer is required: a generic Gaussian rasterizer can
lack the kernel_size/subpixel_offset interface. Native FlashAttention absence
uses the upstream naive-attention fallback. No weight/architecture changes.

3. DOWNLOADS, PREDICTOR INITIALIZATION AND EXECUTION ORDER
On a clean reproduction checkout only (never overwrite the archived reports):
  mkdir -p runs/gap-first-v01/drop/vendor
  git clone https://github.com/chuanshuogushi/ReconPhys runs/gap-first-v01/drop/vendor/ReconPhys
  git -C runs/gap-first-v01/drop/vendor/ReconPhys checkout 184b53f93d6f75d445e3197af52eae48c2492ad6
  python scripts/gap_first_v01/drop_download.py
Compare resulting file hashes against the archived downloads.json, rather than
assuming URLs ending in /main remain immutable.

Download config.json, preprocessor_config.json, configuration_intern_vit.py,
modeling_intern_vit.py and flash_attention.py from the InternViT repository
into runs/gap-first-v01/drop/internvit. Compare their hashes to protocol.json.
Create the native backbone initialization from the SAME released checkpoint:
  python - <<'PY'
  from pathlib import Path
  import torch
  p=Path('runs/gap-first-v01/drop')
  s=torch.load(p/'ckpt.pt',map_location='cpu',weights_only=False)['model']
  b={k[len('backbone.'):]:v for k,v in s.items() if k.startswith('backbone.')}
  assert b
  torch.save(b,p/'internvit/pytorch_model.bin')
  PY
The full checkpoint is subsequently loaded strictly by drop_predict.py. This
extraction does not replace or retrain the backbone. Byte hashes of a newly
serialized torch file may differ; preserve the archived file for byte-exact
verification, or record the regenerated tensor identity in a NEW protocol.

Set execution environment (local device numbering):
  export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
  export PYTHONPATH="$PWD/runs/gap-first-v01/drop/vendor/ReconPhys:$PWD"
  python -c 'from pytorch3d.ops import knn_points; from trellis.utils import render_utils'
  python scripts/gap_first_v01/drop_predict.py
  python scripts/gap_first_v01/drop_run.py --asset 00e2a405dcff4701a8f57d2e67e306b8 --pilot

The original freeze consumed environment.json, native-source-audit.json and
PROTOCOL-DRAFT.txt, native predictor reports, the four-frame pilot, generated
anchor caches and cameras. For a NEW campaign, record the actual environment
and update only a copied campaign's namespace/deadline before freezing:
  python scripts/gap_first_v01/drop_freeze.py
  python scripts/gap_first_v01/drop_supervisor.py
This executes every 3x9 case, then analysis, native-video metrics and the
original-camera figure. To audit archived outputs without rerunning:
  python scripts/gap_first_v01/drop_evidence_audit.py
The supervisor deadline is 2026-11-07T21:44:00+00:00. After that date it refuses
new computation. Never bypass it by editing the archived frozen sources;
create a separately named reproduction protocol and disclose changes.
Completed cases are not overwritten; incomplete cases remain explicit failures.

4. IMMUTABILITY, NATIVE COMPONENTS AND MEASUREMENTS
Primary protocol.json SHA256:
b4d1fda9e101faa8b48b11ce7991604283b2475883d354770c1d127b01d2efd2
Its frozen_sha256 table is authoritative for scientific sources, native source,
configs, assets, checkpoint, predictor outputs, initial states and cameras.
drop_protocol.verify() checks these before runs/analysis. Preserve all frozen
files. The native Spring_Mass_simplify solver, 100 substeps/frame, hard-floor
contact, Gaussian binding/interpolation and GaussianRenderer are imported
unchanged. ForceAdapter ONLY adds a known vector per anchor in compute_force.
The separately released Spring_Mass_simplify_force implementation is NOT used;
it changes binding/interpolation and would confound this comparison.

A/B mass, stiffness and damping are (1,300,1)/(2,600,2). Force is uniformly
applied to all 2048 anchors: horizontal +4 for calibration, vertical +20 for
held response, frames12..17 inclusive. Mass and force are per anchor, units
are normalized native scene units, not calibrated SI; this is a BODY FORCE,
not a localized contact probe or independent constitutive-law test.
Observation: camera0 foreground-centroid horizontal shift at frame18 relative
to passive, maxRGB>8 mask, divided by512; total delta noise sigma=.002.
Target: centroid xyz over frames18..24, 21 scalar coordinates; coordinate RMS
tolerance=.01. This differs by sqrt(3) from point-Euclidean RMS.
Balanced two-world prior, 100 noise repeats/world/asset; 600 noise realizations
are NOT 600 physical objects. Same-evidence MAP is refitted after calibration.

5. SEPARATELY FROZEN NAMED-NATIVE EXTENSION
native-reference-extension/protocol.json SHA256:
9fe5845b7483c87530a550281b6cb3f66db7522408e1533d4b2aabaf00778a56
This posthoc extension was declared and frozen after the primary experiment.
Its script and all inputs have separate hashes; no primary sources changed.
On an empty extension output namespace, after all primary outputs exist:
  python scripts/gap_first_v01/drop_native_extension.py
  python scripts/gap_first_v01/drop_extension_audit.py
It creates lossless RGB movies of passive A/B, feeds native first20 frames to
the unchanged checkpoint, freezes A's inferred parameters before changed-force
replay, and scores that named prediction against the two candidate responses.
A/B numerical sensitivity is reported, not assumed exactly zero. The native
predictor is NOT refitted after calibration; the corrected MAP is separate.
The extension has no silent rerun: existing status/results are retained.
Its manifest was generated before execution; copy it only with identical inputs,
or freeze a new explicitly posthoc reproduction with new hashes/deadline.

6. CAMERAS, FINAL FIGURES AND OUTPUTS
Quantitative camera: native four yaw views, radius2, FOV40, 512x512. Original
camera can clip near the floor. All PSNR/SSIM and corrective observations remain
on these frozen cameras, including masked/full/union metrics in reports.
Presentation camera v1 (illustration-camera.json) still clipped upward motion;
it is superseded, NOT a final figure and NOT a quantitative measurement camera.
Final v2: illustration-camera-v2.json, same camera for every asset/condition,
FOV65.60447, fitted to union bounds of ALL passive/changed frames and assets.
No reference photographs exist at this wider camera; no new-view PSNR claimed.
  python scripts/gap_first_v01/drop_render_illustration_v2.py
  python scripts/gap_first_v01/drop_visualize.py --wide
  python scripts/gap_first_v01/drop_teaser.py
  python scripts/gap_first_v01/drop_figure_audit.py
Arial font files must be installed; plotting scripts register local Arial from
/home/bayespc2/.local/share/fonts/arial. On another host point this NONFROZEN
presentation setting to licensed local Arial fonts and record the new hash.

Final deliverables in output/gap-first-v01/drop/:
  native-drop-wide-comparison.{png,pdf,mp4}: all3 assets, final v2 camera.
  native-drop-teaser.{png,pdf}: first predeclared asset, one common crop over
    all four conditions and all30 frames; no per-panel recentering/rescaling.
  final-figure-audit.json; illustration-render-v2-manifest.json; teaser and
    wide-comparison manifests. All360 frames pass foreground-border check.
Snapshots are passive0.54s / forced0.72s. Video playback is slowed3.3x.
The trace is a projected MODEL control centroid, not an observed surface marker.
Final PDFs embed Arial/Arial-Bold and were rasterized and visually inspected.

Reports: per-asset analysis, preobservation bank, native-video metrics, summary,
primary independent-evidence-audit.json, extension independent-audit.json.
Raw local cases: runs/gap-first-v01/drop/cases/<asset>/<condition>/ contains
result.json, trajectory.npz and Gaussian position arrays. Raw assets, videos,
checkpoints, envs, vendor trees and multi-GB trajectories are LOCAL ONLY.
Paper section: papers/cvpr2027-joint-identification/sections/gap-first-drop.tex.

7. PUBLICATION INCLUDE / EXCLUDE
Include all frozen scientific scripts and quantiles.py; download/predict/
freeze/supervisor helpers; both independent scientific auditors; final v2
renderer, visualize, teaser and figure-audit source. Include all immutable
protocols, environment/source audit, download hashes, actual metrics/summary/
preobservation reports, final audit JSONs, final camera-v2 JSON and final
figures/video/manifests. Include this README and the paper section. Include
original source hashes even when upstream source/data itself is not bundled.

Exclude from the presentation package: illustration-v1/ images, original
clipped native-drop-comparison figure/video, illustration-camera.json (v1),
old drop_render_illustration.py and drop_illustration_supervisor.py outputs,
development-predictor/, pilot imagery, transient logs, locks, process IDs,
compiled binaries and caches. Keep those locally as historical provenance.
The v2 camera records the old camera hash; mention that historical v1 is not
bundled. drop_build.py/drop_readiness.py are machine-specific operational
history, not required portable reproduction scripts; this README replaces them.
Do NOT exclude failed outcomes or negative metric results. No raw upstream
weights/dataset/vendor code redistribution is authorized by this package.
