# Independent nonlinear dynamics: component implementation and mechanics evidence

At the component-test checkpoint, new files were `experiments/campaign_nonlinear_dynamics.py` and
`tests/test_campaign_nonlinear_dynamics.py`. Existing static/native capture
sources remain untouched. The initial component checkpoint preceded any TRAIN case. The later single smoke is recorded below. The independent four-BC gate remains unpassed.

## Declared discrete model

`NonlinearDynamics` uses the reviewed float64 log-J neo-Hookean energy and
exact finite rigid-patch transforms. Both authored handle masses/full COM
inertias are represented by six auxiliary masses each, including an optional
COM offset. Deformable density-volume-lumped masses remain separately present.
All free nodes/rigid poses are solved together. Exact eliminated support nodes
cover a point pin, fixed patch, two fixed patches and a prescribed translated
moving patch. No penalty stiffness, inversion repair or contact is added.

At each step the objective is the body and auxiliary mass-weighted squared
distance to the inertial prediction divided by `2 h²`, plus nonlinear elastic
energy, plus body-only viscous incremental potential, minus held world-force
work at the handle origins and any explicitly supplied nodal loads. Finite
rotation increments compose on the left and are reset after each step. The
preconditioner caches rest stiffness plus a physical first-state mass metric
per timestep. Its approximation does not replace any nonlinear force or
inertial term. Physical mass makes free/pinned dynamic modes nonsingular.

Stored point velocities are unprojected **finite-map chords**, not
instantaneous rigid angular/tangent velocities. To avoid cancellation from
absolute world offsets, the algebraically identical objective uses reduced
position differences and finite-map increments; reduced positions are retained
across steps. The discrepancy between those increments and subtraction of the
stored absolute point positions is explicitly recorded as representation
roundoff. This fixes a micro-force test that otherwise stalled around
3.15e-12 N against a 1e-12 N residual target; the test tolerance was not relaxed.
Elastic energy still uses absolute point positions. This residual check does
not validate arbitrarily small elastic strains below position precision.

Acceptance checks the exact nonlinear gradient with physical world torques
converted using the SO(3) Jacobian, and uses a physical Newton-valued load norm
plus an explicit absolute force floor. No normalization by absolute-coordinate
mass RHS is used. Positive-J line search and iteration caps raise preserved
errors rather than producing a replacement solution.

The step ledger records all point momenta, energy, force/work, full fixture
reactions including inertia/drag, body viscous loss, implicit velocity-increment
loss, elastic-gradient time defect and curved-constraint chord work. Drag is
included in both linear and angular momentum balance. Its algebraic energy
identity is a bookkeeping check, not independent proof of physical accuracy.
Finite-step energy/angular momentum conservation is explicitly not claimed.

## Tests on the current source

The new 10 mechanics tests and 21 relevant inertia/static prerequisite tests
all pass: **31 passed in 17.78 s**, CPU limited to two threads. The suite covers:

- Rest, exact fixed/pin/two-patch constraints and valid dynamics with singular
  static rigid modes; positive-J/hardware/state/cap failures.
- Free rigid and deformable translation under held forces: independent
  momentum and implicit-Euler position references.
- Finite common rotation/translation covariance with transformed full inertia,
  exact rigid distances and nonzero COM-offset kinetic/dynamic momentum checks.
- Small-amplitude nonlinear motion against a separately implemented linear
  implicit-Euler recurrence, retaining the same discrete integration convention.
- Moving-support work/reactions with independently calculated mass acceleration
  and viscous force, plus an independently integrated damped free translation.
- Asymmetric torque-free rigid spin at three timesteps, measuring rather than
  concealing energy and chord-angular-momentum drift.

For the 0.4 s free-spin control:

| Timestep | Relative energy drift | Relative chord-angular-momentum drift |
|---|---:|---:|
| 0.04 s | 6.438% | 3.288% |
| 0.02 s | 3.432% | 1.728% |
| 0.01 s | 1.773% | 0.886% |

Both errors contract by approximately one half under timestep halving. Their
nonzero values are a limitation of this first-order constrained-point scheme.
These component tests do not qualify the four actual TRAIN layouts, the
120/240/480 Hz trajectory gate, residual-tolerance gate, new data generation,
native Isaac behavior, direct neural prediction, uncertainty or probing.

JUnit evidence and frozen source/dependency/gate copies are under
`runs/operator-dev/nonlinear-dynamics-mechanics-v01/`; `validation.json`
records their SHA-256 hashes and all unpassed campaign gates. The first full TRAIN smoke followed publication of the independently predeclared gate and implementation review, as recorded below.

## Subsequent single authorized smoke

After gate publication at main `71d8db9011d21992464aa0b2c7d24c77d4a7ec45`, the
new `campaign_nonlinear_dynamics_case.py` runner and five additional protocol/
input tests passed. Its public `--inputs-json` route regenerates the four
privileged TRAIN static inputs, verifies source hashes, canonical array
dtype/shape/hashes and exact boundary specs, and needs no bulk dataset.

Only t0001 / 120 Hz / diagonal / relative tolerance 1e-7 was then integrated.
Its 360 steps completed in 28.18 s, producing 31 common-time node-position,
chord-velocity and rigid-pose frames. Mean nonlinear iterations were 2.58,
maximum 3; every step met its recorded residual tolerance. Coordinate RMS was
72.782 µm, minimum J 0.998407, fixed-node error zero and maximum rigid patch
distance error 1.25e-16 m. The force impulse error was 2.60e-18 N·s.

Frozen protocol, sources/inputs, full-step residual/work/energy ledger and
trajectory are in
`runs/gaussian-campaign-v01/independent-dynamic-t0001-hz120-diagonal-v01/`.
The separately saved post-run hash/metric audit is
`runs/operator-dev/independent-dynamic-t0001-hz120-smoke-audit-v01.json`.
At this smoke checkpoint no other TRAIN cell had run. Later cells retain their own execution ledgers. Temporal/tolerance refinement, all-support
coverage and reset replay remain incomplete; this smoke qualifies no data,
native behavior, physical inference or probing policy.

## Reproduce without the bulk dataset

The public static fixture contains TRAIN recipes and known hardware, not motion
labels or reserved observations. From the repository root, with NumPy and
PyTorch installed:

```bash
python -m pytest -q tests/test_campaign_nonlinear_dynamics.py tests/test_campaign_nonlinear_dynamics_case.py tests/test_campaign_nonlinear_fem.py tests/test_campaign_rigid_inertia.py tests/test_campaign_linear_fem.py
python experiments/campaign_nonlinear_dynamics_case.py --output runs/my-independent-smoke --inputs-json configs/gaussian_independent_diagnostic_inputs_v01.json --episode t0001 --hz 120 --response diagonal_pulse_handle0 --relative-tolerance 1e-7
```

The complete targeted test command passed **36 tests** in the publication
verification. Outputs require a fresh directory and retain the frozen protocol,
all source/input snapshots, full step ledger, 31 trajectory states, hash-bound
summary and terminal execution state. Failed integrations retain failure
metadata and their last state. The runner restricts inputs to the four declared
TRAIN layouts, initial declared rates, responses, tolerances and translations.

The [published smoke audit](../examples/gaussian-neural-boundary/campaign-diagnosis/independent-dynamic-smoke-v01.json)
records original local artifact hashes. The full trajectory and step ledger are
local evidence pending the next artifact archive; those hashes do not imply
that the raw files have been remotely published.

The [independent dynamic gate](../configs/gaussian_campaign_independent_dynamic_gate_v01.json)
remains separate from the failed native reliability gate and all prediction,
physical inference, calibration and active-probing acceptance requirements.

## Scope of the initial force check

The gate's diagonal peak is `[0.01, 0.01, 0.005]` N at one handle.
Original dataset recipes use 0.12–0.24 N peaks, with a factor of 0.25 for a
point pin, and include push, pull, oscillating shear and a two-handle couple.
Even a passing initial gate establishes accuracy only for its declared inputs.
Before generating replacement data at the broader original force envelope,
TRAIN engineering checks must establish temporal, residual and coordinate
robustness for the actual force profiles/amplitudes and resulting deformation.
A small-force qualification cannot substitute for that evidence.

The [separate absolute-floor study](../configs/gaussian_campaign_dynamic_floor_refinement_v01.json)
addresses an identified accuracy-test limitation: without external force, all
nominal relative tolerances use the same absolute threshold, even when a
moving support drives the body. Its two additional cells use the identical
physical model, reuse the original middle-floor baseline and retain failures.
Nominally different but effectively identical thresholds are not presented as
independent accuracy evidence.

## Initial full-response temporal results

All 24 case/response/rate cells completed and passed the separate protocol,
clock, force, residual, positive-volume and fixture/patch audits. Their
full-coordinate temporal differences (120→240 then 240→480 Hz) are:

| Support | Zero-response differences (µm) | Diagonal differences (µm) | Tolerance (µm) | Initial result |
|---|---:|---:|---:|---|
| Point pin | 0, 0 | 2.315, 1.162 | 119.005 | Pass |
| Fixed patch | 0, 0 | 3.890, 2.159 | 3.781 | Pass |
| Two patches | 0, 0 | 0.237, 0.193 | 2.000 | Pass |
| Moving patch | 16.520, 8.474 | 16.082, 8.262 | 6.507 | **Fail** |

The moving-support errors contract but remain above the unchanged threshold.
The [conditional 960 Hz protocol](../configs/gaussian_campaign_dynamic_temporal_extension_v01.json)
uses the original gate's optional refinement and preserves the failed initial
result. It has not run at this checkpoint. A passing temporal extension alone
will not qualify untested accuracy/translation/replay interactions at 960 Hz.
The remaining factorized 480 Hz accuracy, translation and replay controls are
ongoing. The whole independent reference gate remains **false**.

[Compact initial evidence](../examples/gaussian-neural-boundary/campaign-diagnosis/independent-dynamic-initial-temporal-v01.json)
and the [strict audit implementation](../experiments/campaign_independent_dynamic_audit.py)
retain missing/failed cells rather than converting them into success. The
initial trigger audit is frozen; later results use new reports.
