# Nonlinear posterior audit and sparse-view rerun

**All 36 cases pass the complete finite-chain convergence screen.** The previous
neural disadvantage largely disappears after replacing a path MAP/local Gaussian
with a sampled posterior average. Constant and neural noise are effectively tied;
these data do not establish a neural reconstruction advantage.

Open `examples/converged-posterior/report.html`. Equations are in
`POSTERIOR_INFERENCE.md`; frozen settings and chronological sampler amendments
are in `POSTERIOR_CONVERGENCE_PROTOCOL.md`. Historical reports remain unchanged.

## What was audited

The original experiment used one maximum-density trajectory as its point estimate.
Stationarity of this optimizer was never a posterior convergence test. Its local
Gaussian importance weights collapsed, and an exact-target pCN run failed all
1,548 monitored quantities: worst Rhat 1.302, minimum bulk/tail ESS 10.5/18.3.
One chain remained in a different damping region. Only 2–11 independent refreshes
were accepted per chain, out of 180–226 attempts.

The physical density itself retains state-dependent Q, its log determinant, the
force-volume term and the centered-drag density Jacobian. Existing exact force
identities, normalized-density checks, independent AD/finite differences and
observed-curvature checks passed. The audit found no basis for removing these
terms or treating a local Gaussian as the nonlinear posterior.

## What changed

Gaussian-reference split Hamiltonian proposals replace the random-walk sampler.
The Gaussian part is integrated exactly. Exact mechanics gradients and a local
image Taylor model generate reversible, volume-preserving proposals; full
nonlinear images and the original normalized density decide acceptance. Thus
the Taylor approximation changes computation, not the invariant target.

Four chains per case use independent initializations/momenta. Step size adapts
for 500 discarded transitions, then freezes. Retained budgets are 2,000 per
chain, extended to 4,000 for `stochastic__4102__floor`. No other case needs an
extension. All 296,000 retained states enter estimates and diagnostics. The
force family and every physical/image setting are unchanged.

The suite has **94 passing numerical tests**, including known-distribution
recovery, exact Gaussian rotations in 1,448 dimensions, reversibility,
deliberately inaccurate proposal forces with full-target correction, physical
Taylor value/gradient/Hessian checks, invalid-state rejection and bitwise
continuation against an uninterrupted frozen chain. CI also passes all earlier
smoke pipelines.

## Convergence evidence

Across 36 cases, all **56,172 monitored scalar quantities** pass:

- Maximum rank-normalized/folded split Rhat: **1.00934** (<1.01).
- Minimum bulk ESS: **820.9** (>=400).
- Minimum tail ESS: **555.8** (>=400).
- No frozen chains.

Each floor case monitors 1,549 quantities; constant/neural cases monitor 1,566.
These include every latent coordinate, 84 missing-gap coordinates, log target,
15 conditional force means, profiled quadratic, and applicable force variances
and covariance normalizers. Analytically constant floor-model variances and
normalizers are excluded explicitly; artificial independent force resampling
cannot conceal poor trajectory mixing.

Retained proposals have 30 invalid intermediates and 97 large energy errors
(absolute error >100), concentrated in the floor control: 29/95 floor, 1/2
constant, 0/0 neural. Invalid paths are rejected with correct probability; no
samples are silently replaced. A follow-up independently seeded smaller-step
replication checks the development floor/neural cases, retaining its own full
diagnostics and proposal counts. Finite diagnostics and numerical agreement
cannot prove absence of unvisited modes or physical-model correctness.

Both initial smaller-step replicas pass every convergence monitor and have zero
invalid/large-energy proposals. Floor gap RMSE is 2.684 mm versus primary
2.674 mm. Neural gap RMSE is 2.247 mm versus primary 2.172 mm. A stricter
batch-MCSE check initially finds two correlated coordinates (node 2's vertical
position at adjacent gap times) just beyond an approximate simultaneous cutoff:
maximum |z| **3.553**, threshold **3.434**; largest mean difference **0.332 mm**.
This borderline result is preserved, not relabelled as a pass.

One fixed extension of **both unchanged neural kernels** to 4,000 retained
draws per chain resolves that stricter check: **0/84 exceed the same cutoff**,
maximum |z| **3.157**, largest gap-coordinate mean difference **0.198 mm**.
Extended primary/smaller-step gap RMSE is **2.179/2.221 mm**. Both pass all
monitors (worst Rhat **1.00214/1.00151**, minimum bulk ESS **3145/3644**, tail
ESS **2671/2938**) with zero invalid/large-energy proposals. Cached nonlinear
targets reproduce within **5.5e-12**, and complete original chain prefixes,
target values and proposal-error records are preserved bitwise. These are
supplementary numerical sensitivity checks, not replacements for the frozen
36-case benchmark. Batch-MCSE normal approximations are not calibration proofs.

The maximum reported gap-component batch-mean MCSE across the 36 runs is
**0.124 mm**. This is numerical uncertainty in an estimated posterior mean,
separate from posterior uncertainty in the unknown motion. CPU sampling/final
summaries total approximately 3.73 worker-hours; the longest case takes 9.06
minutes, excluding environment installation/artifact transfers. Workers execute
independently in parallel; these are not GPU training timings.

## Same sparse views and independent reference data

The comparison repeats the original four conditions × three paired seeds ×
three models. Inputs remain six 24×24 RGB images at indices [0,3,6,14,17,20]
from one moving camera; the seven-frame gap is [7..13]. All models use the
known eight-node tetrahedral scene, fixed cameras/appearance/supports, six-step
staggered dynamics, frozen diffusion calibration, and exact external-force
marginalization. Original observed RGB hashes and normalized MAP objectives
reproduce before sampling. Reference trajectories remain the independent
accurate deterministic/stochastic data used previously.

Mean missing-gap coordinate RMSE, **mm**:

| Truth condition | Floor MAP → posterior | Constant MAP → posterior | Neural MAP → posterior |
|---|---:|---:|---:|
| No load | 2.815 → 2.663 | 2.421 → 2.337 | 3.604 → 2.297 |
| External pulse | 2.824 → 2.843 | 2.660 → 2.646 | 3.450 → 2.662 |
| Internal noise | 4.778 → 4.942 | 3.142 → 3.205 | 3.625 → 3.177 |
| Pulse + internal noise | 4.916 → 5.108 | 3.420 → 3.493 | 3.622 → 3.483 |
| Both stochastic conditions pooled | **4.847 → 5.025** | **3.281 → 3.349** | **3.623 → 3.330** |

The neural average improves by about **8.1%** on stochastic cases; its apparent
0.343 mm disadvantage to constant noise becomes a negligible 0.019 mm difference
in the other direction. This small difference is below the reported per-case
Monte Carlo error scale and is supported by only three paired underlying seeds.
**Do not claim a neural-specific advantage.**

The paired neural-minus-constant differences averaged within each underlying
seed are -0.018, -0.066 and +0.027 mm; there are only three independent seed
groups. No ranking test or calibration claim is based on counting correlated
coordinates as independent replicates.

Neural posterior averaging worsens error in four of six stochastic cases;
two large improvements drive the average.
Posterior means optimize expected squared error under the assumed model, not
every individual realization under model discrepancy.

Mean total-force RMSE over the same stochastic cases is **0.01938 / 0.01378 /
0.01373 N** for floor/constant/neural. Force covariance is E[S(z)] + Cov[m(z)],
computed over every retained state. The spatial force profile is still fixed;
this is not arbitrary contact-force localization. This GP represents the unknown
applied load, excluding known gravity and support reactions. The constant
control fixes the internal-noise amplitude while retaining geometry-dependent
edge covariance; it is already a physically structured SDE.

## Why better calibration worsened the old MAP

A controlled calibration-only ablation holds the final mechanics, renderer,
kernel, observations and optimizer fixed. Three pulse-plus-noise seeds give:

| Diffusion amplitude model | Mean MAP gap error (mm) |
|---|---:|
| Legacy constant calibration | 3.066 |
| Legacy neural calibration | 3.160 |
| Calibrated constant | 3.420 |
| Calibrated neural | 3.622 |
| Exact analytic oracle amplitude | 3.606 |

The remaining network amplitude error is not the explanation: replacing it by
the exact simulator law changes mean error by only 0.017 mm (largest case change
0.040 mm). Unlike the historical old/new report comparison, this ablation
isolates calibration under identical final numerical operators.

Correcting the old overestimated amplitude tightened the specified dynamics,
and realistic state dependence changed path-density volume. A high-density
complete path can differ from the region carrying most probability mass.
The normalized density has 1,440 velocity coordinates and includes both
covariance and drag Jacobians; a single centered-path MAP is sensitive to these
volume effects. Dropping the terms would give a different density, not fix
Bayesian inference. Restricted force basis, independent diffusion floor,
discretization and other prior assumptions still contribute model mismatch.

On the declared pulse-plus-noise development case, neural MAP gap error is
4.211 mm and posterior-mean error is 2.172 mm. Re-optimizing from the posterior
mean to decrement <0.001 returns to essentially the original mode: objective
1610.291050 → 1610.290951, error 4.211 → 4.231 mm. This rules out the loose
stopping tolerance or this alternative start as the explanation, while not
excluding every possible remote mode.

At the mean normalized path, image NLL improves by 1.178 and the profiled
dynamic quadratic improves by 0.623 relative to the original MAP. Half logdet Q
increases by 2.020, the negative drag log-Jacobian by 1.680, and half logdet A
by only 0.0034. The mean path has higher overall NLL by 1.878 despite better
reconstruction. This illustrates the density-volume tradeoff; the force-volume
term is small in this particular example. All terms and directional derivatives
are saved in `mode-audit.json`. The sampled mean physical damping is 0.503/s
versus MAP 0.760/s and truth 0.550/s (the exponential of the mean log damping
is a different quantity and is labelled separately in the audit).

The sampled expectation of half the profiled dynamic quadratic is about **715**,
versus **5.25** at the MAP. This is a high-dimensional typical-set effect: many
nonzero innovation paths carry posterior mass even though each has lower density
than the mode. Their average is a summary, not a typical individual realization.
Use joint posterior draws for motion generation.

## Uncertainty results and limits

On the six stochastic benchmark cases, unscaled central component 90% intervals:

| Model | MAP/local coverage → posterior coverage | Local width → posterior width (mm) |
|---|---:|---:|
| Floor | 67.86% → 66.07% | 9.51 → 9.21 |
| Constant | 95.63% → 95.83% | 12.18 → 12.85 |
| Neural | 89.48% → 96.63% | 11.95 → 12.84 |

The nonlinear neural intervals are wider and conservative on these cases.
Coordinates, timestamps and the two stochastic conditions are correlated.
These frequencies do not establish repeated-realization Bayesian calibration.
The earlier 19-motion conformal calibration / 20-fresh-motion simultaneous-gap
experiment remains a separate prediction-set result and is not rerun or used
to scale these posterior bands. The present benchmark was already inspected;
there is no new blind-test or real-video claim.

## Reproduction and handover

Model/inference source: `6a00d90c0ca8a53183c8346fc28def808c8e26ec`.
Numerical source fingerprint:
`7d5b5c85b57a45819136acb4657b8b7c5041c0e145de12da0fb6683331984c27`.
Later script/docs commits preserve this fingerprint. Python 3.12.4, CPU Torch
2.14.1, NumPy 2.0.0, opt-einsum 3.4.0, two Torch threads per worker. The fixed
original spatial projection prevents tied-eigenspace backend ambiguity.

```sh
pip install -e '.[performance]'
python -m unittest discover -s tests -v
python scripts/converged_sparse_comparison.py --case-index 29 --output runs/new-posterior
# Repeat --case-index 0..35, optionally on independent workers sharing outputs.
python scripts/converged_sparse_comparison.py --merge --output runs/new-posterior
python scripts/check_posterior_step_size.py --case-index 27 --output runs/new-check
python scripts/check_posterior_step_size.py --case-index 29 --output runs/new-check
python scripts/audit_calibration_effect.py --output runs/new-calibration-effect
```

The workflow `posterior-comparison.yml` runs all 36 cases and aggregates compact
JSON results. Its public run is
https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39062694272 .
Independent smaller-step checks are run **39063241871**; the single fixed-budget
neural extension is **39065071522**. The latter retains `extended-chains-primary`
and `extended-chains-small_step`; their initial and extended comparisons are
versioned alongside the primary data. `scripts/compare_posterior_replicas.py`
recomputes batch-MCSE differences and checks cached targets from full states.
Artifacts `posterior-chains-0` … `posterior-chains-35` retain every full latent
draw, fixed Gaussian factor, energy error and RNG checkpoint for 30 days.
The repository permanently retains exact original inputs, complete diagnostics,
all posterior means/intervals, total force covariance components, 400 paired
gap-position/parameter/force draws per case, audit states, plots, CSV and hashes.
Compact draws support gap-view inspection; regenerate full latent states for
full dynamics-conditioned rollouts. They are never independently recombined.

Next scientific steps are posterior predictive checks and repeated-realization
calibration on new seeds, time/mesh and discrepancy-floor sensitivity, then
unknown-scene/camera and real-video experiments. Increasing network capacity
is not supported by a 0.019 mm difference on this benchmark.
