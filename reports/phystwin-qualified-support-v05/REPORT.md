# Qualified reference and probing campaign v05 — in progress

All 12 development objects now have qualified nonlinear references. The final
held-out comparison has not run yet: the finer hypothesis bank is being generated
because the previous bank failed its development accuracy gate.

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

The n=8 bank is complete: 324 physical hypotheses, each evaluated under four loads.
Its predefined extrapolation still misses the 0.03 mm forward-error gate on two
hard objects (0.0304 and 0.0530 mm). The protocol therefore requires the n=12 bank
before model selection can finish. No accuracy threshold was relaxed.

Model selection uses only the 12 development objects. Spare CPUs are preparing
full-face calibration simulations in parallel; those responses do not select the
model. Twenty new calibration objects will determine the prediction-band scale
and fixed-action baseline. The model and calibration choices will then be sealed
before evaluating 24 reserved test objects, with four noise repeats per object.

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
