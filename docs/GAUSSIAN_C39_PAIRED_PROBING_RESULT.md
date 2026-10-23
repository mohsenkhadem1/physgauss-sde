# Downward versus sideways probing: paired development result

Choosing the load direction improves held-load prediction in this controlled experiment. A simple support-based rule makes exactly the same choices, so the experiment does not establish an advantage from sophisticated uncertainty planning.

## Experiment and numerical checks

The experiment reuses C38's four fixed configurations, 16 noise repetitions each, exact gravity-prefix observations, and original downward-poke posteriors. Its design was frozen before C38's private evaluation. It is a paired development extension, not a new population test or temporal forecast. Both candidate probes have magnitude 2 N and acquire nine ideal 3D tracked points after a common-rest reset. This is equal observation count and force magnitude, not equal mechanical work.

The new sideways library passes all 10 state/20 field mechanics checks and fixed q63 spatial indicators (3.069–17.962 micrometres). All eight withheld interpolation checks pass the frozen 1 micrometre gate, maximum error 0.0227 micrometres. Local fixed-edge stretch strain reaches 90.44%; only the specified point displacements are qualified, not stresses, continuum accuracy or real-material validity.

All 128 model action designs pass their positive-cloud checks without fallback. Every choice is sealed before fresh sideways noise is drawn. All 128 new postprobe model executions pass the original numerical criteria, and all workers are reaped before private scoring. No initial or downward observations/predictions are regenerated.

## Prediction results

Pooled coordinate RMSE at the 63 target points under the held +1 N load, in micrometres:

| Model | Gravity prefix | Selected direction | Always down | Always side | Random |
|---|---:|---:|---:|---:|---:|
| Nonlinear table | 23.759 | 10.896 | 12.855 | 13.828 | 14.121 |
| Linear rest tangent | 23.826 | 11.009 | 12.846 | 14.477 | 14.440 |

Both information gain and target-risk selection choose 32 downward and 32 sideways actions under both models. For the nonlinear model, selected-direction mean per-record RMSE is 8.804 micrometres, compared with 18.443 before probing. Its paired mean advantage is:

| Baseline | Mean advantage (µm) | Conditional noise-bootstrap 95% interval (µm) |
|---|---:|---:|
| Always down | 1.400 | [0.064, 2.824] |
| Always side | 1.788 | [0.070, 3.555] |
| Random | 1.780 | [0.036, 3.750] |

These are unadjusted comparisons across noise repetitions within four fixed configurations; the small positive lower limits should not be treated as broad evidence of general superiority. The linear selector's interval against always-down includes zero. The nonlinear selector's mean advantage over the linear selector is only 0.135 micrometres, interval [-0.396, 0.695]; there is no convincing nonlinear advantage here.

## The stronger simple-rule control

An explicitly **posthoc diagnostic** applies this rule using the original public posterior's most likely support category and the preexisting public fine-linear action ranking: down for end clamps, side for separated bottom patches. It reproduces both uncertainty criteria's choices in **64/64 records under both models**. Therefore it has exactly the same predictions and zero performance difference from those criteria. No private labels or actual action outcomes are needed to construct the rule.

This control was not prospectively registered as another policy. It explains the observed result without claiming another independent trial. Prefix support inference was already correct and nearly certain in every record; ambiguous boundaries and useful exploration of unresolved support uncertainty remain untested.

## Material estimates and unresolved uncertainty

For the nonlinear selected policy, mean relative error of the material posterior median decreases from 5.07% to 2.25%; its material 95% interval covers 64/64 records after probing. Vertical pointwise coverage is 576/576. These conditional counts do not establish population calibration.

The old 10-nanometre coordinate-quantile stopping rule remains frozen. Raw full-3D pointwise coverage is 1507/1728 (87.2%), with a demonstrated failure to resolve some tiny nonconstant coordinate distributions. Numerical quantile enclosures cover 1728/1728, but are not substitute 95% credible intervals. The linear model's raw coverage is 1368/1728 and its numerical-enclosure coverage 1506/1728. See `GAUSSIAN_QUANTILE_RESOLUTION_DIAGNOSIS.md` for the separate numerical repair; no completed comparison was overwritten or posthoc widened.

The next numerical diagnostic holds the same likelihood, priors, physical hypotheses and observations fixed, checks bit identity of normalizers/material estimates/moments, and refines only coordinate quantiles in separate outputs. Remaining disagreement with the direct forward reference must then be distinguished from root resolution. Drop/contact trajectories remain unqualified, and no neural/Gaussian/RGB-D or future-motion superiority is claimed.

Full scalar evidence: `runs/gaussian-campaign-v01/c39-full-evaluation-v01/summary.json`; compact result and posthoc-rule check: `runs/operator-dev/c39-sealed-result-report-v01/compact-result.json`. Raw fields and per-record private metric arrays remain local.
