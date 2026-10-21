# Official PIDG feasibility refresh — 21 October 2026

The latest official commit is still `b77fd2107a50c875d7219bb4f82388d5e5ea1319` (26 February 2026). Its clean checkout still lacks the imported material module. **The official historical source is recoverable and matches the released bytecode under the recorded code fingerprint.** This removes the missing-source dead end, but does not establish a working build or an official training result. No packages were installed, upstream modules executed, training started, or GPU simulation launched during this refresh.

## Recover the official implementation, without inventing a replacement

The latest [Gaussian model](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/scene_PIDG/gaussian_model.py) imports `motion_utils.time_evolving_material_field`; the complete recursive tree contains only its CPython 3.8 bytecode. Official [historical commit 907cd641](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/907cd641ea01bf46a4c03ab509e6d4872c78af6b/motion_utils/time_evolving_material_field.py) contains the source. It was removed in [053ad223](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/commit/053ad223fdffd9674902a34c485f95b24086fdc1). The closed missing-file issue does not supersede the actual latest tree.

| Artifact | SHA256 |
|---|---|
| Historical official source, 9,561 bytes | `1b9c97a6602a4831f7f134f5d9c01063cceedf91c51fc8116b17acc3912215cb` |
| Latest released `.pyc`, 7,949 bytes | `a2aedd99c36dd36fc1695f77b4e19e4ca857dbb496e5bc3ba4aeaa1b95eac166` |
| Equal normalized module-code fingerprint | `c7d09b597b9d31a5a2be8c24063c2cfdebe1b01a7545fbf16d0c85d88a4eba0b` |

`historical-full-module-comparison.json` records comparison of the entire compiled historical module against the released bytecode, using the already-installed CPython 3.8.20 interpreter. Our own `inspect_bytecode.py` uses `compile` and `marshal.loads`; it never imports or executes the upstream module. The fingerprint includes recursive bytecode, constants, names, variable/free/cell names, argument counts and flags. It excludes debug locations, stacksize and nlocals metadata. Equality is an exact match for those fields, not a runtime or trained-model equivalence claim. Synthetic tests verify ignored debug locations and detection of changed defaults/nested code.

Simply aliasing the other `VelocityNet` in `motion_utils/pidg.py` would be a different restoration: its default maximum particle count is 100,000 versus 1,200,000, and it lacks the historical save/load methods. A future isolated external checkout should restore the pinned historical official file and record the one-file restoration patch. Source was inspected only under `/tmp/pidg-source-audit-imuci680`; this project retains metadata and our inspection script, not upstream implementation. Latest repository license metadata is null and the tree has no license candidate; dependency terms remain separate.

## Build feasibility: plausible, not built

The README's Python 3.7.16 conflicts with its pinned Torch 2.5 generation; the project's Python 3.12 conflicts with pinned NumPy 1.21.6 / SciPy 1.7.3. **Propose an isolated Python 3.10 environment**, Torch 2.5.0+cu121, matching CUDA 12.1 compiler toolkit, a supported GCC11/C++17 toolchain and Ninja. Python 3.10 has the relevant published NumPy/SciPy and official cu121 Torch wheels. The cu121 index has no Torch 2.5 CPython3.8 Linux wheel; recovered source removes that bytecode-only constraint. This is a proposed dependency intersection, not a fully resolved or installed lock. Old MMCV/YAPF and the actual training import closure still need a build smoke test.

This machine has four RTX2080Ti GPUs (SM7.5), driver 595.91.07, and project Torch metadata 2.14.1. `nvcc`, C/C++ compilers, Ninja/Make/CMake and Conda were absent from PATH at audit time. A CUDA driver/runtime is insufficient for import-time hashencoder JIT, simple-knn and rasterizer CUDA builds. The latest tree has no `.gitmodules`; explicitly check out the [flow rasterizer](https://github.com/HaoqinHong/flow-based-diff-gaussian-rasterization/tree/2146a1c0dc3558644af420f20c21a0ad07ffa67a) at gitlink `2146a1c0dc3558644af420f20c21a0ad07ffa67a`. Do not install the upstream full requirements into our shared environment. Evidence: `local-toolchain.json`, `package-compatibility.json`, `build-dependency-evidence.json`.

## Fair comparison on a future independent dataset

Official PIDG is a per-scene coordinate/time reconstruction model. It is a relevant held-view/gap-filling baseline. It has no query force/boundary input; predicting a new force is not an official action-conditioned capability. Same-action temporal extrapolation may be measured as force-unaware extrapolation. Material posterior calibration and active probing are not supplied by this baseline. Any force-conditioned extension or corrected-physics variant must be separately named. Our existing PIDG-inspired MLP remains explicitly inspired and is never relabelled official.

After the independent dataset passes its numerical gate:

1. Create a separate pinned external checkout/environment; restore the exact official historical file, compile all three CUDA extensions and record versions, source hashes, changes and an import/render smoke test. No training until this passes.
2. Adapt public RGB, camera poses/times, masks and the common allowed rest point cloud to the official synthetic reader. Keep private tetrahedra, materials, fixtures and target trajectories out of fitting. Validate camera convention, scale and nonuniform time gaps.
3. The reader expects `flows_flo`, `motion_mask`, `depth_distill` and forward/backward flow/occlusion files. Compute auxiliary flow only from allowed observed images, never a withheld target; audit the source's adjacent-time logic for sparse gaps. Estimated depth/flow supervision must be matched or explicitly labelled; do not quietly provide metric truth depth to one RGB method.
4. Supply validation cameras to the training loop's `getTestCameras()` checkpoint selector; keep sealed final test views outside fitting, flow generation, checkpoint selection and tuning. Start with one TRAIN/dev scene under a declared budget. Log seeds, failures, optimization time, peak VRAM, Gaussian budget/densification and per-scene fitting cost.
5. Preserve the released momentum residual variant. The prior audit's convective-index counterexample still applies at latest commit; report any corrected-advection variant separately. Compare reconstruction under identical allowed observations, then report changed-force forecasting as a distinct task, without inventing an official PIDG action policy.

The existing sealed results remain frozen. This refresh produces **no official PIDG score**. `audit.json` is the compact finding; `upstream-proof.json`, `source-index.json` and fingerprint outputs establish provenance. `publication-inventory.json` scopes the evidence suitable for publication without copying upstream implementation.
