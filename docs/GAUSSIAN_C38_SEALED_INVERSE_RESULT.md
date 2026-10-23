# Sealed controlled inverse experiment: results

An extra downward measurement improves held-load prediction and material estimation in this controlled static experiment. Uncertainty-based action selection does **not** outperform the strongest fixed action. Full 3D interval calibration remains unresolved.

## What was tested

Four fixed physical configurations (two withheld stiffnesses, two fixed-support layouts), each with 16 independent noise repetitions: **64 records, not 64 independent objects**. Each prefix contains nine ideal tracked 3D top points under gravity. Cartesian noise is stipulated at 200 micrometres; it is not calibrated camera noise. A policy acquires one additional 27-coordinate measurement after a separate static reset, choosing a known downward or upward 2 N load. The prediction target is the displacement of 63 material points under a withheld +1 N load.

The nonlinear model uses a qualified three-stiffness forward table; the baseline uses the linear rest tangent. Both infer unknown stiffness and support category from the same prior. Initial predictions, every policy choice, and all 256 postprobe model executions were sealed before private material/support labels or canonical target values were analyzed. All 128 initial and 256 postprobe model executions passed their frozen numerical checks. Raw observations, fields, labels, and per-record metric arrays remain local.

## Prediction and inference

Pooled coordinate RMSE at the 63 target points, in micrometres:

| Model | Prefix only | Downward poke | Upward poke | Random poke |
|---|---:|---:|---:|---:|
| Nonlinear table | 23.759 | 12.855 | 21.641 | 17.530 |
| Linear rest tangent | 23.826 | 12.846 | 21.787 | 17.511 |

For the nonlinear model, the downward acquisition reduces pooled RMSE by 45.9%. Mean per-record RMSE falls from 18.443 to 10.204 micrometres. The paired mean improvement is 8.239 micrometres, with a conditional noise-bootstrap 95% interval of [4.862, 11.639]. This resamples repetitions within the four fixed configurations; it is not a population interval. The nonlinear model has no convincing mean advantage over the linear baseline: the prefix-only paired advantage is 0.035 micrometres, interval [-0.301, 0.377].

Mean relative error of the nonlinear posterior material median falls from 5.07% to 2.94%. Its material 95% interval covers the truth in 63/64 records both before and after the downward poke. The support category is already correct in all 64 prefixes with near-unit probability. Thus this experiment does not demonstrate inference of genuinely ambiguous boundaries.

## Actions and uncertainty limitations

Both information gain and target prediction risk choose **downward in every record under both models**, with no numerical or Monte Carlo fallback. Their predictions are therefore exactly the fixed-down baseline. Random chooses down 34 times and up 30 times. The experiment establishes a useful additional measurement, not superior intelligent probing.

Nonlinear raw pointwise 95% coverage across all nine points and three axes falls from **1538/1728 (89.0%) to 1483/1728 (85.8%)**. Coverage in the vertical axis is 567/576 (98.4%) both times. Coordinates are correlated; these counts do not establish joint-field or population calibration.

There is a numerical-resolution issue in very narrow horizontal intervals: 166 initial and 221 postprobe misses lie inside the prospectively frozen numerical quantile-root enclosures, with maximum such miss 3.286 nanometres. The root-position resolution was fixed at 10 nanometres before observation generation. The numerical enclosures cover 1704/1728 coordinates both times. **These are numerical enclosures of calculated quantiles, not newly widened 95% credible intervals**; their coverage cannot replace raw coverage. Other misses remain, with maximum raw miss 58.03 micrometres before and 20.12 afterwards. The linear model also has horizontal model discrepancy and worse enclosure coverage after acquisition. No posthoc interval correction was applied.

## What follows

The C39 extension was frozen before this private analysis. It compares downward and sideways loads of equal magnitude, reusing the exact existing prefixes and downward outcomes. Public linear design checks show that the preferred direction changes with support layout, but this is only a design proxy. New sideways nonlinear fields, spatial and interpolation validation, all action-choice seals, and fresh joint posterior checks must pass before its results are scored.

Contact/drop simulations still fail spatial accuracy requirements and remain excluded from the prediction dataset. These results do not demonstrate RGB-D inference, moving Gaussian reconstruction, future motion forecasting, calibrated full physical uncertainty, or a learned predictor outperforming a strong mechanics baseline.

The machine-readable full scalar summary is `runs/gaussian-campaign-v01/c38-full-evaluation-v01/summary.json`; compact counts and provenance are in `runs/operator-dev/c38-full-evaluation-report-v01/compact-result.json`.
