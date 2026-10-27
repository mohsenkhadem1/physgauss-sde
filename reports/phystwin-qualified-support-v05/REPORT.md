# Qualified reference and probing campaign v05 — in progress

All 12 development and 20 calibration objects now have qualified nonlinear references. The completed
finer hypothesis bank passes its development accuracy gate, and posterior
refinement passes at 257×257 parameter nodes per fixture category. All 48 development probe
decisions pass sampling and posterior-grid stability checks. Calibration reference
generation is complete. Calibration inference stopped before sealing because
physical parameter quantiles did not converge at its original grid limit. A
numerical refinement extension is running; the final held-out comparison has not run yet.

## What is verified

The original displacement, strain, support-force, energy and equilibrium gates
remain unchanged. Every accepted reference passes two successive spatial
refinements and a quadrature check under all four loads. Ten development references
use P2 tetrahedra; two difficult half-face fixtures use Q2 hexahedra concentrated
near the support edges. Earlier failed P2 attempts are retained.

The new Q2 method passed an independent comparison against the qualified P2
reference under every load: worst displacement RMS difference 0.00427 mm,
traction difference 0.257%, and energy difference 0.135%. An additional finer
check on the hardest object's initial load passed with RMS difference
0.0000237 mm. [Numerical diagnosis](NUMERICS.md), [validation](tensor-validation.json),
and the [method selection rule](tensor-reference-adoption.json) record the evidence.

These experiments contain measurable deformation. Every development object has
at least one probe with nonrigid coordinate RMS of at least 0.238 mm, and at least
9.4% of motion RMS remains after removing rigid translation and rotation. This is
an available-probe diagnostic, not a claim that every observation or chosen action
is equally informative.

## Why the comparison is still pending

The n=8 and n=12 banks are complete: each contains 324 physical hypotheses evaluated
under four loads. The n=8/n=4 extrapolation failed the 0.03 mm forward-error gate on
two hard objects (0.0304 and 0.0530 mm). The predefined n=12/n=8 extrapolation passes:
maximum forward RMS error is 0.0101 mm, and maximum bank interpolation RMS error is
0.00243 mm. [Model selection](model-selection.json) retains every rejected candidate.
No accuracy threshold was relaxed.

The development posterior required the full 257×257 grid per fixture category.
The final refinement changes predictive means by at most 0.00278 mm, standard
deviations by 0.00236 mm, and physical log-parameter quantiles by 0.00976; all pass
the frozen gates. Earlier grids failed. [Refinement progress](posterior-refinement-progress.json)
records these checks. All 48 development decisions pass sampling and grid checks;
47 of 48 match the strong fixture heuristic. This is development evidence, not a
held-out probing advantage. [Development summary](development-summary.json).

Model selection uses only the 12 development objects. CPUs helped prepare
full-face calibration simulations; those responses did not select the
model. Twenty new calibration objects will determine the prediction-band scale
and fixed-action baseline. The model and calibration choices will then be sealed
before evaluating 24 reserved test objects, with four noise repeats per object.

All 20 calibration simulations have now qualified. Calibration predictions were
stable at the 257×257 grid (maximum mean change 0.00183 mm), but physical
log-parameter quantiles changed by 0.0834, exceeding the unchanged 0.02 limit.
The runner stopped without sealing or opening the test responses. The
[numerical extension](posterior-refinement-extension.json) adds 513×513 and 1025×1025 levels
to the same posterior integration, with bounded likelihood batches to fit GPU
memory. It does not change the model, priors, likelihood or accuracy gates.
[Failed checks](calibration-refinement-progress.json) are retained. The 513-grid quantile change was 0.02095, still above 0.02, so doubling continues
to 1025. Three observation-budget mixtures are distributed over GPUs 0–2, moments
and likelihoods are processed in bounded chunks, and utility grid checks run
sequentially. Twenty-five targeted tests pass, including three-GPU/CPU prediction
and expected-risk parity. Development is being repeated.

The analysis will separately report:

- Posterior numerical convergence, including physical-parameter interval endpoints.
- Held-force prediction error, shape error after rigid alignment, and strain error.
- Raw material/support interval coverage and whether probing creates wrong confidence.
- Adaptive probing versus fixed, random and inferred-fixture-side choices.
- A privileged post-test action diagnostic, clearly separated from eligible policies.

Uncertainty summaries use object-level resampling; noise repeats are not treated
as independent objects. The results video will show median and worst test objects
before and after probing. It depicts camera motion around static held-force shapes,
not a dynamics forecast.

## Execution and limits

The integrated mechanics, inference and failure-handling suite passes 40 tests.
GPU solves use float64, check the original CPU residual, and release the obsolete
fine matrix before assembling its replacement. Memory admission chooses a CPU
fallback when a hierarchy cannot fit. The largest host-memory jobs are serialized.
[Environment](execution-environment.json) and [execution notes](memory-execution-addendum.json)
record the setup. Raw datasets remain local.

The durable runner is `scripts/phystwin/supervise_qualified_campaign.py`; local
status is `runs/phystwin-qualified-support-v05/status/supervisor.json`. It stops on
failed reference, forward-model, deformation or posterior-convergence gates.

This is a static synthetic reference study with known geometry and forces, unknown
material stiffness, support stiffness and four fixture categories. It does not yet
establish general neural boundary inference, learned PhysTwin uncertainty, real
RGB-D performance, gravity/contact prediction, or superiority of active probing.
The numerical checks are finite evidence, not absolute continuum-error bounds.
