# Unknown point-support/material sparse-depth result

Two sparse depth frames improve prediction on one fresh H0 object with unknown material and support node, while the point-pin type, full hardware and known force programs are supplied.

| Coordinate RMSE | Prior mean | Conditioned mean |
|---|---:|---:|
| A, future after observed 1 s prefix | 1.2580 mm | 0.06064 mm |
| B, independent reset/all 61 frames | 0.81935 mm | 0.05437 mm |
| B, after 1 s | 1.01067 mm | 0.06707 mm |

The true mechanical support node 17 rises from 5.36% prior to 84.17% posterior probability, rank 1. Other nodes retain probability; there is no continuous-center or BC-type inference. Equal 2,048-particle prior/posterior predictions improve proper held 72-ray joint mixture log density A 185.766→191.424 and B 192.935→197.632. B resets from rest, without sequential probing.

All 55 exact prior-mass support masks are integrated with material-prior weights p(mask)/Nmask, using 1,024 draws/mask with nested 256 prefixes for three fixed seeds (168,960 queries). Cameras/pixels use rest geometry; only 36 rays at 0.5/1.0 s enter inference. All seeds freeze before truth/held evaluation. Noise 0.2 mm and visibility error 0.005 are stipulated working assumptions, without calibrated physical discrepancy.

Material E remains broad: mean 9,662 Pa, 90% interval 2,834–18,696 Pa, true 7,593 Pa. Density mean 506 kg/m³, interval 435–580, true 522. Poisson ratio and drag remain nearly prior-wide. All four truths lie inside their intervals on one object, without population coverage evidence.

Minimum largest-run ESS 257.37; cross-seed support TV 0.02412; log-evidence range 0.11685; material mean range ≤0.191 posterior SD. Maximum particle-refinement support TV 0.02741. Numerical variation is visible; these diagnostics do not prove convergence.

Four supplied-true controls give full-linear/cached-modal A forecast 41.786/41.791 µm and B all-frame error 19.200/19.209 µm, versus inferred 60.640/54.368 µm. Modal-full whole-trajectory errors 0.573/0.555 µm are small, while linearization/finite-rigidity discrepancy remains a substantial part of residual error. Controls run after freeze without fitting, reference reruns or posterior tuning.

Scope: one point-pin H0 case; no general BCs, subnode centers, SBC, coverage calibration, active policy, RGB encoder, native/continuum claim. Larger-patch geometry extension remains unvalidated and excluded. Raw caches, truth, observations, particles and predictions remain LOCAL.

## Reproduction

Regenerate fresh inputs with prepare_unknown_point_oracle.py, execute its two frozen finite protocols, then build all 55 nominal bases with campaign_support_basis_cache.py prepare/build using generated hashes. See docs/GAUSSIAN_POINT_SUPPORT_ORACLE.md for methods and source-linked evidence. With original local inputs/cache present, replay into a FRESH output:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python experiments/run_campaign_point_material_reference.py \
  --design runs/operator-dev/unknown-point-oracle-inputs-v01/protocol-design.json \
  --cache runs/gaussian-campaign-v01/point-support-bases-v01/cache-manifest.json \
  --cache-sha256 642f50c696129c0cafda781b87a3342d3effcc25ca5484172afb25b604977664 \
  --output runs/gaussian-campaign-v01/unknown-point-oracle-replay-v01 \
  --device cuda:0 --batch-size 512
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python experiments/evaluate_campaign_point_material_reference.py \
  --reference runs/gaussian-campaign-v01/unknown-point-oracle-replay-v01
```
