# Fixed likelihood correction: closer estimates, incomplete numerical qualification

The frozen neural posteriors failed DEV/CAL reference checks. A separate numerical study used them only as proposals, with the exact all-acquired-data likelihood and original uniform intrinsic-logE/logK prior. It did not retrain the networks, change physical parameters or observations, or open TEST.

The first proposal was half prior and half frozen neural density. It stopped after 13.36 s because a Beta inverse CDF rounded a sample to the prior boundary. Three complete cases and a fourth partial case remain preserved. No sample was clipped or redrawn. A proposed shape increment of one also failed an extreme-value **pure control** before any new study. Both preliminary source versions and failure receipts are retained.

The reviewed replacement used the same mixture weights, but incremented each proposal component's two Beta shapes by **two**. Its own normalized density, not the original neural density, appeared in the importance denominator. This changes only a sampling proposal: the target remains `likelihood × original prior`. Half uniform prior preserves full support. All shifted neural proposals were evaluated anew; four completed prior-only cases and two completed evidence-normalizer groups were reused by hash.

Thirty-two DEV/CAL cases were fixed in advance, two per observation context per role. Four independently scrambled Sobol clouds used nested budgets of 1024, 4096, 16384 and 65536 hypotheses, with every result retained. Pure prior QMC had the same hypothesis budgets and common underlying Sobol coordinates. It was computed once per case and compared with all three neural proposals, not counted as three independent baseline runs.

| Proposal | Cases passing at 16384 | Cases passing at 65536 | Worst field-mean error at 65536, µm | Minimum ESS at 65536 |
|---|---:|---:|---:|---:|
| Prior only | 18/32 | 30/32 | 1.426 | 582 |
| Shifted seed 2026122410 | 3/32 | 24/32 | 4.711 | 1408 |
| Shifted seed 2026122411 | 4/32 | 20/32 | 3.709 | 1608 |
| Shifted seed 2026122412 | 4/32 | 18/32 | 3.495 | 1750 |

No proposal passed all cases; none passed the same thresholds at 1024 or 4096. The declared criteria included CDF-grid error ≤0.005, field mean/std error ≤2 µm, relative evidence error ≤0.005, ESS ≥400, maximum normalized weight ≤0.01, and separate between-scramble spread limits. Prior-only sampling passed all absolute error criteria, but two cases failed the stricter between-scramble CDF-spread criterion: maximum spread 0.00646. The neural proposals also failed some moment, evidence, or between-scramble checks. Their larger ESS therefore did not establish better numerical accuracy.

The corrected estimates were much closer to the continuous reference than the uncorrected networks. However, the neural proposals did not beat the simple prior-QMC baseline under this fixed experiment. The nonlinear target, learned density and acquisition observations stayed unchanged; no architecture, proposal fallback or sampling-budget search followed these outcomes.

CDF/field references were reused from the qualified continuous calculation. Evidence was obtained explicitly with the same qualified low/high continuous normalizer calculation on the same observations; it was not inferred from CDF agreement. Added computations and reused groups are recorded separately. The final run completed in 157.71 s on CPU2 with about 654 MB peak RSS and no GPU allocation. Completion is an engineering result; the numerical qualification remained negative.

These are weighted QMC clouds, not MCMC chains or independent posterior samples. Four scrambles and finite-grid/moment agreement would not prove full-joint convergence or population calibration even if every threshold passed. This remains a known-layout, ideal-endpoint-sensor, homogeneous logarithmic-NH static-force-interpolation control. TEST and neural active probing remain closed; the previously qualified [continuous reference](GAUSSIAN_C38_BAR_REFERENCE.md) is unchanged.

See the [original neural negative result](GAUSSIAN_C38_LEARNED_POSTERIOR.md), [original correction protocol](../runs/operator-dev/c38-beta-importance-v01/protocol-v01.json), [shifted proposal protocol](../runs/operator-dev/c38-beta-importance-v02/protocol-v02.json), and [all-resolution compact results](../runs/operator-dev/c38-beta-importance-v02/result-summary.json). Sources are [the original corrector](../experiments/campaign_c38_beta_importance.py) and [the separate proposal repair](../experiments/campaign_c38_beta_importance_shifted.py). Raw archives and detailed cloud records remain LOCAL; reproducing the recorded numerical studies requires their hash-bound local prerequisites and fresh output paths.
