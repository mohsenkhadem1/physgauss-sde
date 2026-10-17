# Simulation accuracy and observation coverage audit

## Findings

Force formulas, discrete momentum balances and camera conventions pass the
audit. Additional observations improve local information and often improve
reconstruction. However, the historical time step has about **3.3 mm coordinate
error** against converged independent RK4, comparable to our reconstruction
errors. Passing implementation tests and numerical stability were insufficient
to establish millimetre-level continuous-time physical accuracy.

Three separate limitations matter: sparse observations, integration error, and
the renderer's previously measured depth-order discontinuities. More views
cannot resolve all three. Historical results remain useful numerical inverse
problems under the declared discretization; they do not validate physical
identification or calibrated uncertainty against an accurate physical reference.

## Actual images

Open the [offline audit report](../examples/simulation-view-audit/report.html).
The [input/reconstruction sheet](../examples/simulation-view-audit/input_reconstruction.png)
shows exact published noisy inputs, a missing timestamp, simulation truth,
the previous marginalized MAP and RGB error. Native images are **24x24**,
displayed with nearest-neighbor enlargement. No high-resolution detail was
generated. These are a coloured 30-Gaussian volumetric proxy with known rest
shape and appearance, not real video or unknown-geometry reconstruction.

The [unseen-view sheet](../examples/simulation-view-audit/unseen_views.png)
compares four directions at the same missing timestamp. The
[mesh/camera figure](../examples/simulation-view-audit/mesh_camera_coverage.png)
shows the actual eight-node, six-tetrahedron solid and supports at true scale.
Display RGB is clipped to [0,1]; inference uses original unclipped values.

## Independent mechanical and camera checks

- Twelve perturbed states: force versus an independently written scalar
  Neo-Hookean energy differentiated with autograd. Maximum force discrepancy
  6.66e-16 N; energy discrepancy 2.50e-16 J.
- All four original truth conditions, every solver step: free-node momentum
  residual below 7.17e-18 Ns; angular-momentum residual including support
  reactions below 1.87e-17 kg m^2/s. Supports exchange momentum with the object.
- Internal stochastic impulses: net below 8.68e-19 Ns and pre-step net torque
  impulse below 4.34e-19 N m s. Support displacement/velocity are zero.
- Minimum det(F) over original logged solver states: 0.9084; no hidden inversions.
- Camera orthogonality error below 4.45e-16; look-at principal-point error below
  3.56e-15 pixels. All centers have positive depth. No center or projected
  three-standard-deviation contour leaves the audited mixed-sequence images.
- New tests verify multi-camera pose-block information against full image AD,
  and single-camera information against the existing trajectory implementation.

Constants remain a prescribed toy model: E=19.5 Pa, rho=1 kg/m^3, nu=0.3,
gravity=0.6 m/s^2, drag=0.55/s, side=0.65 m and mass=0.274625 kg. These are not
experimentally measured materials. Mesh convergence, contact/plasticity and
fluctuation-dissipation validation have not been established.

## Time integration accuracy

A separate classical RK4 routine integrates the mechanical ODE with the same
constitutive law. Its 48-versus-96-substep coordinate difference is about
2.1e-10 m, supporting temporal convergence for these two deterministic cases.
This is not spatial mesh convergence or experimental validation.

| Euler substeps per 0.05 s frame | dt [ms] | No-load error vs RK4 [mm] | Pulse error [mm] |
|---|---:|---:|---:|
| 3, historical setting | 16.667 | 3.292 | 3.290 |
| 6 | 8.333 | 1.549 | 1.548 |
| 12 | 4.167 | 0.753 | 0.753 |
| 24 | 2.083 | 0.372 | 0.371 |
| 48 | 1.042 | 0.185 | 0.184 |

Errors average all frames and free-node coordinates; maximum historical node
error reaches **11.5 mm**. The rest-state linearized damped Euler stability
limit is about 83.60 ms. The historical step is stable under that criterion,
but insufficiently accurate. No-load work-energy defect decreases from
2.047 mJ at three substeps to 0.0664 mJ at 48. Drag dissipates energy; exact
energy conservation is not imposed on this first-order integrator.

Brownian-bridge refinement preserves original coarse increments in both
stochastic cases. Error relative to the coupled 48-substep path decreases
from about 3.12 mm (three substeps) to 0.191 mm (24 substeps). This is one
coupled realization, not an exact stochastic solution or measured ensemble
convergence rate. At fixed state, 8,192 innovations confirm impulse covariance
proportional to dt, with 2.59% empirical covariance error in Frobenius norm.

Original data and inference share the coarse integrator, concealing continuous
integration bias. Sub-millimetre reconstruction of that discretized truth does
not imply sub-millimetre physical accuracy. Two new
[high-accuracy deterministic reference datasets](../examples/simulation-view-audit/high_accuracy_reference/)
provide converged RK4 positions/velocities and newly rendered six-image inputs
with the same original noise realization. They were not used in historical
fits or the view ablation. An absolute-accuracy regression test requires the
audited 48-substep no-load Euler case to stay below 0.25 mm against RK4.
Historical configurations remain preserved for reproducibility.

## Temporal versus spatial observations

The original camera covers -25 to 75 degrees azimuth at one elevation. Its
horizontal lens field of view is 33.72 degrees; 100 degrees describes the orbit.
Observed times are 0, 0.15, 0.30, 0.70, 0.85 and 1.0 s. The 0.4 s gap hides
the force peak.

Ten marginal-MAP fits use two original sequences and five observation schemes.
All retain exact original six-image arrays, fixed force/discrepancy priors,
the same six-image-only initial state and 20 maximum iterations. Extra images
use independent fixed noise seeds. Secondary camera azimuth is offset by 100
degrees. Twelve single-camera times and six simultaneous two-camera pairs
both supply twelve images. The twelve-time indices are
[0,2,3,5,6,8,10,12,14,17,18,20].

| Observations | Images | Original-gap pose error, pulse [mm] | Original-gap pose error, mixed [mm] |
|---|---:|---:|---:|
| 6 times, 1 camera | 6 | 1.639 | 4.454* |
| 12 times, 1 camera | 12 | 1.072 | 2.563* |
| 6 times, 2 cameras | 12 | 1.245 | 3.107* |
| 21 times, 1 camera | 21 | 0.801 | 2.336* |
| 21 times, 2 cameras | 42 | 0.494* | 1.945 |

An asterisk denotes local stationarity (GN decrement <0.05); only 5/10 fits
pass. Other endpoints hit the declared line-search limit. These are finite
optimization outcomes on one geometry/seed, not reliable rankings or calibrated
posterior comparisons. The mixed dense two-camera fit has the lowest gap error
but fails stationarity, so its apparent advantage is especially provisional.

The metric evaluates the seven times originally inside the gap. They become
observed in dense schedules: this is **not held-out temporal error** there.
JSON additionally reports all-frame error, still-unobserved-time error where
applicable, and a third evaluation camera at +200 degrees that supplies no inputs.

For this gap metric, twelve single-camera times beat six two-camera pairs in
both sequences. Other metrics differ: two-camera pairs have lower all-frame
error in these runs, and lower force error on pure pulse truth. No universal
winner follows. Local information accumulation is algebraically identical to
the original RGB-Jacobian calculation; floating-point differences can change
finite trajectories at depth ties, so exact old iterates need not be reproduced.

## Information and identifiability

At an identical reference state, image-only local rank is 63 for six times,
135 for twelve and 243 for 21, versus 728 exposed trajectory/material
coordinates. A second camera retains the same maximum pose dimension per time
while improving conditioning. Dynamics and priors supply essential information;
dimension counting alone does not prove nonidentifiability of the constrained
problem. Known geometry and appearance strongly simplify these sensitivity tests.

Pulse-case standardized force covariance trace falls from 5.453 to 3.431
(12 single-camera times), 3.399 (six two-camera pairs), 2.952 (21 single-camera
times) and 1.231 (21 two-camera pairs). The mixed case follows the same pattern.
State/model/noise are held fixed for this local Gaussian information comparison;
the publication validator checks covariance decreases with added observations.
This is not calibrated nonlinear force uncertainty.

## Next work and reproduction

Use converged reference inputs to test inference separately from data-generation
integration. Improve the inference integrator and visibility objective before
physical identification or calibration claims. Simply increasing current
all-at-once inference to 48 substeps creates 11,528 state coordinates: sparse
linear algebra or a correctly derived higher-order/collocation density is
needed. Changing integrators can change density Jacobians and force-elimination
assumptions. For the intended one-moving-camera setting, denser observations
through the event are useful; the second camera is a diagnostic information
bound. Unknown geometry/appearance/cameras, more independent seeds and real
video remain future work.

Audit implementation: `38be8cb237e2a6e937794f92c254536cacd78015`.
Raw full audit: `runs/simulation-view-audit-final`. Manifest records script
hash, package fingerprint, environment and historical input hash. Exact added
images and ten fitted paths are published; large Gaussian checkpoints stay local.

```sh
python scripts/audit_simulation_views.py --output runs/new-simulation-view-audit
python scripts/publish_simulation_audit.py --raw runs/new-simulation-view-audit \
  --output examples/new-simulation-view-audit --source-ref 38be8cb
physgauss check
```

Use a fresh output directory. Stage flags `--images-only`, `--physics-only`
and `--views-only` support inspection; publication requires a full run. Source
changes during view fitting stop execution. Publication checks source,
shared input/initial-state hashes, mechanical/camera gates, time accuracy,
all ten records and added-information covariance. Text hashes normalize line
endings; binary hashes use raw bytes. Figures were inspected visually.
