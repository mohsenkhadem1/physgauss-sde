# Official PIDG environment and derivative smoke — 21 October 2026

**The official implementation now builds, imports and renders on this machine in an isolated external environment. The renderer gradient correctness control fails when its flow flag is enabled.** This is environment/runtime evidence, not an official trained benchmark or a physics validation. No dataset was read and no optimizer/training step was performed.

## Isolation and provenance

All upstream code, binaries, package caches and raw logs remain outside this repository at:

`/home/bayespc2/.local/share/physgauss-external/pidg-official-v01`

The external checkout pins official main `b77fd2107a50c875d7219bb4f82388d5e5ea1319`. Its only implementation change is restoration of `motion_utils/time_evolving_material_field.py` from official commit `907cd641ea01bf46a4c03ab509e6d4872c78af6b`, SHA256 `1b9c97a6602a4831f7f134f5d9c01063cceedf91c51fc8116b17acc3912215cb`. The previous feasibility audit proves its normalized CPython3.8 module-code fingerprint matches the released bytecode. The original pinned Git blobs and restoration patch remain external. The flow rasterizer is the official gitlink `2146a1c0dc3558644af420f20c21a0ad07ffa67a`. No tracked Python/CUDA/C++ implementation was edited; builds/imports regenerated some tracked bytecode and egg-info metadata, recorded in `source-provenance.json`.

The project `.venv`, shared packages, system packages and shell configuration were untouched. Available disk/RAM were approximately161/103GiB at start. The completed external tree occupies9,403,641,253 apparent bytes (~8.76GiB), leaving162,669,912,064 disk bytes (~151.5GiB) free. Build concurrency was capped at `MAX_JOBS=2`; GPU0/3 remained hidden and only reserved GPU1 was used for two short synthetic diagnostics.

## What actually works

The isolated prefix has Python3.10.22, Torch2.5.0+cu121, Torchvision0.20.0+cu121, nvcc12.1.105, GCC11.4.0 and Ninja; NumPy1.21.6/SciPy1.7.3 match upstream pins. A minimal actual train/render/metrics import closure was installed rather than the mixed full requirements list. Exact Conda URLs, pip versions, compiler versions, commands and artifact hashes are retained here; no claim is made that the original full dependency list was solved.

- Official simple-knn and pinned flow rasterizer wheels built successfully; the official hashencoder JIT build also succeeded.
- All eight imports pass: native nearest-neighbor extension, flow rasterizer, hashencoder, scene module, renderer, `train_pidg`, `render`, and `metrics`. `pip check` passes.
- Nine synthetic Gaussians render at32×32 through the official GaussianModel/renderer, with all nine visible. RGB output is finite and nonzero. Backward gradients for positions, SH color, opacity, scaling and rotation are finite with flow flag false and true.
- Hashencoder forward and first/second backward checks are finite. The restored official velocity/stress network returns finite9×3 and9×6 tensors.
- The first GPU smoke used87,811,072 peak allocated /102,760,448 peak reserved Torch bytes. Both diagnostic processes exited0; their PIDs are gone and GPU1 has no compute processes (10MiB idle context accounting). Exit0 for the derivative diagnostic means measurement completed; its correctness gate is explicitly **false**.

Preserved failures and repairs: direct native-extension import initially lacked Torch's preloaded symbols, so our own harness now imports Torch first, as official modules do. Hashencoder initially lacked search paths to the already-installed NVIDIA math headers; isolated CPATH/CPLUS_INCLUDE_PATH now include those directories. Actual train/render imports additionally required upstream-pinned Matplotlib3.5.3. All failures, commands and full logs remain external and are hash-indexed here. No upstream compatibility or scientific code patch was needed.

## Finite-difference counterexample: enabled flow flag doubles a geometry component

A second and final bounded GPU1 process used the same fixed nine Gaussians/32×32 camera, a pure mean-squared RGB loss against black, no flow objective, cleared gradients and restored parameters on every call. Four declared position/log-scale coordinates were checked at every central-difference step0.003,0.001,0.0003; all24 comparisons are retained in `derivative-control.json`. Forward images are bitwise equal across flow flags.

For the central Gaussian's x log-scale, analytical/finite-difference ratios are:

| Central-difference step | Flow flag false | Flow flag true |
|---|---:|---:|
| 0.003 | 1.00005 | 2.00011 |
| 0.001 | 1.00017 | 2.00033 |
| 0.0003 | 0.99891 | 1.99782 |

Fine-step x-position and edge-scale derivatives likewise give approximately1 versus2. Coarser steps disagree for some coordinates; these failures are retained. All Gaussians have identical z=2, so moving a single z changes depth sorting at a tie: that z finite-difference sequence is not interpreted as a smooth derivative reference. This control does not validate every other derivative.

Read-only [pinned rasterizer source](https://github.com/HaoqinHong/flow-based-diff-gaussian-rasterization/blob/2146a1c0dc3558644af420f20c21a0ad07ffa67a/cuda_rasterizer/backward.cu#L638) explains the stable counterexample: normal RGB backward accumulates projected-mean/conic gradients; when the flow flag is true, lines658–666 also add the same RGB contributions into auxiliary incoming-gradient buffers. Covariance/mean preprocessing then sums both paths (lines215–218 and408–427). This double counts the tested photometric geometry component. Appearance, SH-direction and auxiliary flow terms have other paths; do not generalize the exact factor2 to the complete training gradient.

The issue is relevant to a real training code path, not just an unused API: [official train_pidg.py136–144](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting/blob/b77fd2107a50c875d7219bb4f82388d5e5ea1319/train_pidg.py#L136) enables the main RGB renderer's flag when `use_flow` is true, iteration is divisible by100 and warmup is over. Auxiliary flow renders also enable it. The default is false, but several released scene configurations enable it; their values/hashes are recorded in `derivative-source-analysis.json`. No training was run, so trained-model effects remain unmeasured. The released implementation is preserved; any future correction must be a separately labelled variant.

## Reuse and remaining work

Our published helper scripts are inspection/reproduction records; use their originals under the external root, where they resolve the pinned checkout and prefix. For a read-only import check using the isolated compiler/header paths, from the external `upstream` directory:

```sh
python3 ../run_logged.py new-import-check ../env/bin/python -c "import train_pidg, render, metrics"
```

The runner defaults to no visible GPU, permits only explicit GPU1 for this scope, sets the isolated compiler/header/cache paths and saves bounded command/log artifacts. Use a fresh log name. The original smoke helpers write fixed metadata filenames: preserve their existing output directory before any replay. A future GPU replay must separately confirm GPU1 availability and coordination.

No further jobs are queued. Training is still untested and independent dataset adapters/evaluation remain to be built after the physics gate passes. Official PIDG supplies reconstruction/gap filling, with per-scene fitting and its allowed preprocessing reported; it has no changed-force query input or calibrated material posterior/probing policy. Existing sealed tests and PIDG-inspired results remain frozen and distinctly labelled. Future official results should preserve the released variant and separately identify any residual/rasterizer corrections.

`publication-inventory.json` scopes metadata and our own helper scripts only. Upstream implementation, restoration patch, compiled extensions, wheel files and raw logs are intentionally excluded from repository publication; their exact external paths/hashes are retained.
