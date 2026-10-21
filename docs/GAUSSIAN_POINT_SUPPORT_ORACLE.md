# Sparse-depth point-support and material oracle

Two sparse depth frames improve deformation prediction on one fresh object
whose point-support node and material are unknown. The point-pin boundary type,
reference mesh, full rigid hardware and force programs are supplied. This is a
numerical working-model posterior, without an image encoder or learned inverse.

| Coordinate RMSE against finite mechanics | Prior mean | Conditioned mean |
|---|---:|---:|
| Program A, future after the observed 1 s prefix | 1.2580 mm | 0.06064 mm |
| Program B, independent reset, all 61 frames | 0.81935 mm | 0.05437 mm |
| Program B, after 1 s | 1.01067 mm | 0.06707 mm |

The true mechanical support node, 17, rises from 5.36% prior probability to
84.17% posterior probability and ranks first. Other nodes retain probability;
this is not certain localization or inference of a continuous subnode center.
Both predictions use equal 2,048-particle resamples from the fixed integration
library. Program B starts from rest; no continuing-state poke or action policy
is tested.

The held sensor score evaluates the proper joint mixture density of the same
72 noisy depth measurements for each predictor. Higher is better: A improves
185.766→191.424 and B 192.935→197.632. These one-case scores do not establish
uncertainty calibration.

## What is integrated

Public H0 hardware is fixed before a fresh full-prior draw, conditional only on
the known point-pin type. No support node, material or response is selected for
success. The geometry prior exactly pushes the normalized support atlas onto
55 mechanically distinct node masks through 473 positive regions. Its total
mass is exactly one. Continuous centers remain proper conditional nuisance
regions because this mesh's mechanics depends only on the selected node.

Each mask receives 1,024 independent material-prior draws per seed; its first
256 draws form a nested refinement. Every sample carries prior integration
weight p(mask)/N_mask, rather than equal weight across masks. Three fixed seeds
integrate 168,960 material/support particles. Material marginals are E
log-uniform and density, Poisson ratio and body drag uniform in physical units.
No category is dropped or renormalized after observing data.

Only 36 depth rays at each of 0.5 and 1.0 s enter inference. Cameras and pixels
are chosen from public rest geometry. Missing rays remain censored. The working
sensor assumes independent 0.2 mm depth noise and visibility-error probability
0.005. Its forward discrepancy is not physically calibrated. All reference
seeds freeze before opening physical labels or evaluating held A/B motion.
The first reference seed, resampling seeds and held queries are fixed in advance.
The 1.5/2.5 s held depth queries use another prescribed camera view.

## Material ambiguity and numerical variation

First fixed reference seed, nominal 90% intervals:

| Material | Posterior mean | Interval | True value |
|---|---:|---:|---:|
| E | 9,662 Pa | 2,834–18,696 Pa | 7,593 Pa |
| Density | 506 kg/m³ | 435–580 kg/m³ | 522 kg/m³ |
| Poisson ratio | 0.346 | 0.287–0.410 | 0.369 |
| Body drag | 0.585 s⁻¹ | 0.367–0.815 s⁻¹ | 0.515 s⁻¹ |

Density is better constrained; stiffness remains broad, and Poisson ratio and
drag remain close to their prior uncertainty. All four true values fall inside
their intervals on this one object. That does not establish population coverage.
Accurate deformation prediction here coexists with ambiguous materials.

The smallest largest-run particle ESS is 257.37. Across independent seeds,
support total variation reaches 0.02412, log-evidence range 0.11685 and material
mean range 0.191 posterior SD. The largest 256→1,024 support refinement shift is
0.02741 in total variation. Monte Carlo variation remains visible; these
diagnostics alone do not prove posterior convergence or physical calibration.

## Forward error remains part of the explanation

Four predictions with the supplied true support/material separate reduction
and linearization error from inference ambiguity:

| Aligned target | Full linear | Cached modal | Conditioned prediction |
|---|---:|---:|---:|
| A, future after 1 s | 41.786 µm | 41.791 µm | 60.640 µm |
| B, all 61 frames | 19.200 µm | 19.209 µm | 54.368 µm |

Modal versus full-linear whole-trajectory RMSE is only 0.573/0.555 µm for A/B.
Thus reduction error is small here, while the linearized mechanical model
accounts for a visible part of the remaining finite-target error. Its
infinitesimal rigid rotations have distance defects. No native or continuum
equivalence follows. These four controls were run after the posterior froze;
they neither refit the posterior nor select its budget.

This result covers one H0 object and a known point-pin type. Larger support
patch geometry extensions remain unvalidated and are excluded. General BC
type inference, continuous support centers, calibrated model discrepancy,
population coverage, SBC and useful active probing remain unfinished.

## Evidence and reproduction

- [Frozen evaluation](../runs/gaussian-campaign-v01/unknown-point-oracle-v01/joint-reference-v01/evaluation.json)
- [Seed/count diagnostics](../runs/gaussian-campaign-v01/unknown-point-oracle-v01/joint-reference-v01/reference-stability.json)
- [True-parameter forward controls](../runs/operator-dev/unknown-point-forward-controls-v01/diagnosis.json)
- [Prospective input and reference design](../runs/operator-dev/unknown-point-oracle-inputs-v01/protocol-design.json)

Raw truth, observations, tensor caches, particles and predictions remain local.
On a new checkout, regenerate inputs with `prepare_unknown_point_oracle.py`
using fresh `--output` and `--finite-prefix` directories; execute its two pinned
finite protocols, then construct all 55 bases with
`campaign_support_basis_cache.py prepare` and `build`. Use the generated artifact
paths and hashes. Existing runs are never overwritten.

With the original local inputs/cache available, the frozen reference settings
can be replayed into a fresh output directory:

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
