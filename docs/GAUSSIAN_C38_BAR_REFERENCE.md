# C38 small 3D material/support reference

**Completed: all 64 objects, eight reference groups and 32 independent integration checks pass the frozen numerical gates. Active criteria do not outperform the simple opposite-endpoint rule.** This is a small continuous inverse benchmark while broader forward-data qualification proceeds. It does not replace that qualification or demonstrate inference of general Gaussian boundary layouts.

The reference is a homogeneous 0.24 × 0.12 × 0.08 m prism with compressible **logarithmic** Neo-Hookean energy

`W(F) = mu/2 * (tr(F.T F) - 3) - mu*log(J) + lambda/2 * log(J)^2`.

This is the immutable C37 bar law, distinct from the new GPU block's polynomial stable-NH law. A known uniform spring acts only along the left normal, with unknown total stiffness K. Exact transverse symmetry rollers remove rigid freedoms without constraining homogeneous lateral contraction. The right force is a known uniform reference-area dead traction. Density is fixed and does not enter this static equilibrium.

The exact compatible field is `q = (Fload/K + s*Xx, t*Xy, t*Xz)`, where the nonlinear axial and lateral stretches solve the original constitutive equations. The [new P2 assembly](../experiments/campaign_c38_bar_reference.py) independently uses the consistent six-node left-face spring Gram and right-face load integrals. Tests check full-field equilibrium, energy, reactions and an independent sparse Newton solve from rest, plus zero-force/prior-boundary parity with the separate Brent oracle. The new constructor explicitly authors the oracle's geometry: blindly using the old block constructor would use a different width. That mismatch was found and corrected in unit development before any study observations.

## Frozen scientific design

There are 64 independent log-prior objects: `E in [4500,9000] Pa`, `K in [100,400] N/m`, with nu=0.35. One prospectively random left/right endpoint provides an x-displacement measurement under 1 N. This is an **idealized known point measurement**, not an image, rendered depth, reconstructed point or physical sensor demonstration. Each measurement has independent known Gaussian standard deviation 200 µm.

The three candidate acquisitions are left endpoint at 1 N, right endpoint at 1 N, and a 2 N probe at the prefix endpoint. Each probe resets from rest. Every policy subsequently predicts the same canonical **1.5 N static load interpolation** from the same rest state. It is not a temporal forecast. One frame/one measurement/200 µm noise and a nominal 0.2 s acquisition budget are equal across actions. The duration is an idealized equal-cost accounting assumption; dynamic settling and mechanical action feasibility are not demonstrated.

The six fixed policies are left, right, random, opposite endpoint, information gain, and predictive risk. The latter two use 64 common-random-number fantasies; score and paired-advantage Monte Carlo standard errors are retained. Integration convergence does not prove that the selected action ranking has converged. Actual noise is fixed per object/action and shared by policies selecting that action.

The [runner](../experiments/run_campaign_c38_bar_active.py) freezes every object's choice before synthesizing any action outcome. All four posterior states (prefix and each of three actions) freeze before canonical target and material-label evaluation. No parameter/noise/control tuning or object filtering is allowed. Whole-field predictive risk uses exact positive 2 × 2 × 2 uniform-volume quadrature for this affine deformation. All raw study arrays remain local.

## Numerical reference and preserved attempts

[Continuous integration](../experiments/campaign_c38_bar_posterior.py) analytically identifies the Gaussian conditional likelihood in `c=1/K`, retains its proper `1/c` prior and full finite prior boundaries, and integrates in truncated-normal probability coordinates. The outer intrinsic log-E integral is adaptive. Two tolerances and conditional quadrature orders must agree in marginal CDFs, parameter/response quantiles, normalizers and predictive moments. Direct independent adaptive 2D cubature checks eight prospectively fixed objects in all four states. The design's quadrature atoms must reproduce continuous predictive moments before selecting an action.

The first invocation failed on relative/absolute protocol-path bookkeeping **before generating any objects or observations**; its empty output, source/protocol and error receipt are preserved. The unchanged source was invoked with an absolute protocol path and fresh output.

That scientific attempt then failed its original predictive-standard-deviation agreement gate: 0.200895 µm versus a 0.200 µm limit after 158.709 s. Its other first-prefix checks passed, including CDF disagreement 5.22e-6 and held quantile disagreement 14.7 nm. No choices, action outcomes or canonical targets were evaluated. This entire failed attempt remains a failure.

The [explicit continuation](../experiments/continue_campaign_c38_bar_active_v03.py) reuses the exact saved 64 objects and prefix observations by hash, changes conditional quadrature orders from 32/64 to 64/128, and keeps all scientific parameters, noise, seeds, outer tolerances and acceptance criteria unchanged. Its wall cap is 1800 s on CPU with two numerical threads. Low/high group summaries are checkpointed before subsequent work. A [separate exact cache](../experiments/campaign_c38_bar_posterior_cached.py) memoizes repeated scalar adaptive nodes; bitwise parity tests cover boundary and two-observation cases. It performs no interpolation or forward-model approximation.

## Completed results

The final continuation completed in 1048.784 s with 618 MB peak RSS; it reuses the earlier completed calculations. The authorized handoff and earlier failed attempt have separate retained costs and outcomes. Maximum two-resolution disagreement is 1.19e-6 in marginal CDF, 0.00161 Pa / 0.000286 N/m in E/K quantiles, 4.73 nm in held-response quantiles, 9.80 nm in field means, and 52.5 nm in field standard deviations. Successful numerical integration is distinct from useful action selection or physical calibration.

| Policy | Common 1.5 N target, volume-weighted coordinate RMSE |
|---|---:|
| Prefix posterior, no new action | 0.39309 mm |
| Fixed left | 0.31754 mm |
| Fixed right | 0.27618 mm |
| Random | 0.33478 mm |
| Opposite-endpoint heuristic | **0.15519 mm** |
| Information gain | 0.16500 mm |
| Predictive risk | 0.15591 mm |

Information gain and predictive risk improve over fixed/random choices in the descriptive paired-object MSE intervals. They provide **no advantage over the opposite-endpoint rule**: risk minus heuristic has mean 2.24e-10 m² and 95% paired-bootstrap interval [0,6.69e-10] m²; information gain is slightly worse ([9.90e-11,7.38e-9] m²). Risk chooses the heuristic action on 62/64 objects; information gain on 55/64. These intervals are descriptive comparisons, without a familywise multiple-comparison claim. The 64-fantasy information-gain top-two gap is within two paired Monte Carlo standard errors on seven objects; numerical reference agreement is not a ranking-convergence proof.

Under predictive risk, 90% E/K intervals cover 58/64 and 53/64 objects: 90.6% [80.7%,96.5%] and82.8% [71.3%,91.1%], respectively. Mean E/K interval widths are 1417 Pa and 33.6 N/m. Held left/right intervals cover 53/64 and 59/64, with mean widths 0.921/0.942 mm. All policies' exact binomial intervals, widths and marginal truth-CDF histograms are retained in the compact summary. Each policy has 64 independent objects; repeated policies/coordinates are not extra independent calibration samples. Coverage has no forced 90% pass criterion, and 64 objects give wide intervals.

This is a prior-predictive check inside one stipulated numerical model, not evidence of real-world Bayesian calibration, RGB-D inference, population validity for unknown layouts, learned Gaussian uncertainty or generally intelligent probing. The previous single-noise bar misses, failed four-dimensional grids and old sealed studies remain unchanged.

The order64/128 continuation subsequently passed both prefix groups and all three left-prefix post-action groups. An explicitly authorized resource handoff stopped it with `KeyboardInterrupt` after 887.248 s, preserving the terminal partial state and completed checkpoints. At that point all 64 choices and all action observations were already persisted, and canonical targets had not been evaluated.

The [checkpoint continuation](../experiments/continue_campaign_c38_bar_active_v04.py) copies those exact choice bytes and observations, reuses qualified summaries and independent checks, and computes only the three remaining right-prefix groups. Its [new inner integrator](../experiments/campaign_c38_bar_posterior_z.py) uses ordinary Gauss–Legendre nodes in standardized Gaussian coordinates. It retains full finite prior mass and checks the exact omitted Gaussian mass beyond a 50-log-density drop; multiplying by the `1/c` prior's maximum ratio of four must remain below 1e-14. Quadrature error is separately tested by the same orders64/128 and independent 2D checks. Synthetic checks cover outside-center tails, prior boundaries and conditional moments, and agree with the previous high-order field moments within 20 nm. This is a numerical implementation change, not a new observation cohort, revised policy or relaxed scientific threshold.


## Evidence and replay

[Compact results](../runs/operator-dev/c38-bar-active-reference-v01/result-summary-v04.json), [preserved attempts](../runs/operator-dev/c38-bar-active-reference-v01/attempts.json), and [terminal/source proof](../runs/operator-dev/c38-bar-active-reference-v01/terminal-verification-v04.json) bind the completed result. The final protocol is [v04](../runs/operator-dev/c38-bar-active-reference-v01/protocol-v04.json). Source/test/protocol and compact reports are publishable; objects, observations, choice files and detailed posterior fields remain local.

A new replay requires the already-created LOCAL artifacts at their pinned hashes and a fresh protocol/output; it is not a one-command fresh-clone reproduction. The command below records the completed invocation, using an absolute protocol path:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 .venv/bin/python -m experiments.continue_campaign_c38_bar_active_v04 run \
 --protocol /absolute/repository/runs/operator-dev/c38-bar-active-reference-v01/protocol-v04.json \
 --protocol-sha256 981e3a5bb9bb3ba1fe1a258f844988de401a9ae1857a800cd1b438dcadc37efe
```

The frozen output path already exists and is intentionally not overwritten. The original order 32/64 scientific attempt intentionally fails the retained standard-deviation gate; the initial relative-path invocation fails before data generation. Neither failure is relabeled as a successful run.
