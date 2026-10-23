# A numerical limit of the old coordinate uncertainty bands

The C38 sealed mean-prediction and material results remain as reported. Its raw coordinate “95%” bands are not always resolved well enough to interpret their nominal probability. C39 retains the same frozen numerical method for a reproducible comparison; a separate prospective method addresses this issue.

## Diagnosis without truth access

The posthoc diagnostic reads only sealed posterior outputs, never private material/support labels, canonical target fields, or measurement arrays. It checks all 128 initial and 128 downward postprobe model outputs. Both independent integration methods agree on the finding.

For the nonlinear model, among 64×27 coordinate intervals, **154 initial and 221 postprobe intervals have zero width even though their polynomial target distribution is nonconstant**. Among the corresponding 5,184 continuous quantiles, 994 initial and 1,091 postprobe returned midpoint CDF values differ from their requested probability by more than 0.01. The maximum discrepancy is 0.975. All position brackets still satisfy the frozen 10-nanometre absolute tolerance.

The failure mechanism is straightforward: if an entire coordinate distribution is narrower than the positional stopping tolerance, the same bracket midpoint can be returned for its lower, median, and upper quantiles. Independent integration agreement does not fix a shared stopping-rule limitation. The published numerical root enclosures retain a different interpretation from their midpoint estimates; their coverage must not be substituted for credible-interval calibration.

This diagnostic does not revise the underlying posterior, add variance, or change the completed results. Material quantiles use a separate log-stiffness stopping rule. Mean predictions and the action criteria use integrated moments rather than these coordinate quantile midpoints.

## Prospective numerical repair

`experiments/campaign_resolved_quantiles.py` refines continuous coordinate quantiles until **both** conditions hold:

- The position bracket is no wider than the smaller of 10 nanometres and 10⁻⁸ times the original polynomial target range.
- The numerical CDF bounds at the returned value lie within 10⁻⁶ of the requested probability.

Constant and mixture atoms keep explicit atom handling. Unresolved CDF uncertainty or exhausted precision remains a failure. This changes numerical resolution, not the physical model, likelihood, prior, observations, interval probability, or interval width by an added margin.

Nine manufactured controls pass, including tiny uniform, log-uniform inverse-stiffness and nonmonotone quadratic distributions, numerical-error failure, and the actual polynomial posterior under adaptive, GL64 and GL128 integration. Ordinary and 10⁻¹²-scale targets have analytically known quantiles. No actual C38/C39 prediction or policy has yet been replaced by this method; any subsequent diagnostic rerun must have separate versioned outputs and cannot be called a new population test.

Diagnostic scalars: `runs/operator-dev/c38-quantile-resolution-diagnosis-v01/summary.json`. The source and observations of the frozen C39 comparison remain unchanged.
