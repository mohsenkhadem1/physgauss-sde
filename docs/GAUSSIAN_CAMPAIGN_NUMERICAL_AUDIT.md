# Numerical reference audit for the recovery campaign

Known-force material calibration remains blocked by numerical sensitivity of the
current Isaac reference. These are TRAIN diagnostics, not a new predictive
comparison. No reserved control gate, calibration response, or fresh
confirmation response was used predictively. No existing simulator, forward-model or
training source was modified by these experiments.

## What the completed experiments establish

The measured forces combine as intended, stationary zero-force responses are
about 4 nm RMS, and fresh 960 Hz trajectories reproduce the earlier trajectories
bit for bit. This establishes repeatability at one configuration, not numerical
convergence or a physically calibrated constitutive law.

For TRAIN t0001, the unchanged material, supports, mesh, hardware, 31 sample
times and diagonal pulse give:

| Physics rate | Diagonal coordinate RMS from rest | Minimum saved volume ratio |
|---|---:|---:|
| 480 Hz | 111.255 µm | 0.98737 |
| 960 Hz | 236.556 µm | 0.94060 |
| 1920 Hz | 148.370 µm | 0.96569 |

The 480→960 Hz trajectory difference is 181.216 µm coordinate RMSE; the
960→1920 Hz difference increases to 275.427 µm. Maximum nodal differences are
2.917 and 3.937 mm. Force-impulse quadrature agrees with the analytic pulse
within 1.38e-11 N·s. Recorded clocks exactly equal the number of steps times the
representable float32 timestep. The World timestep getter separately returns
the nominal double value; this distinction is recorded rather than concealed.
All original sensor, geometry, support and force checks pass.

The initial base validator reached its final reset check and raised
`KeyError('poke_0')`: the numerical menu names that candidate `zero_force`.
Those failures remain in the execution ledger. A separate, fully saved validator
adapts only its two reset-reference dictionary keys. It retains every original
clock, force, source, geometry, camera, ray, capture-hash and replay check and
every tolerance. No images or physics were regenerated for this adaptation.

## Iteration and coordinate factorial

Six cells were declared before observing their responses: 960 Hz with 32, 64,
128 deformable iterations, in the original coordinates and with every physical,
reference, support, handle and camera coordinate translated by −0.25 m in z.
Local attachment offsets, inertia and camera rays are invariant under this
translation. Rigid solver iterations remain at their composed fallback of 16.

| Condition | Iterations | Diagonal RMS | Status |
|---|---:|---:|---|
| Original | 32 | 236.556 µm | Full independent QC passed |
| Original | 64 | 682.998 µm | Full independent QC passed |
| Original | 128 | — | Compression guard failed at 1.436458 s |
| Recentered | 32 | 39.694 µm | Full independent QC passed |
| Recentered | 64 | 152.797 µm | Full independent QC passed |
| Recentered | 128 | 886.273 µm | Full independent QC passed |

The original 128-iteration failure has minimum tetrahedral volume ratio
0.097525, maximum nodal displacement 27.27 mm and fixture error only 6.7 nm.
It is preserved, never replaced or omitted as an unsuccessful scene. In the
recentered condition, successive iteration differences increase from 136.776
to 838.901 µm RMSE. This is not a converged operating point.

At matched iteration counts, undoing the world translation leaves 233.214 µm
(32 iterations) and 662.680 µm (64 iterations) differences. In this initial
factorial original and recentered conditions ran on GPUs 0 and 3, respectively.
A separately frozen crossed-device 32-iteration replay reproduces each
coordinate condition bitwise after swapping GPUs 0 and 3, including full
trajectory archive hashes for zero, diagonal and reset responses. Assignment
to those two devices therefore does not explain the coordinate difference in
this case. See `numerical-solver-crossed-v01/cross-device-comparison.json`; the
six-cell source and results remain immutable.

Every successful cell has nine initial warmup renders and 93 synchronous
captures with bitwise unchanged observed nodal q/v, rigid poses and velocities,
clocks, step counts and timestep settings around the rendering operation. This
does not expose every hidden warm-start cache, force cache or kinematic flag.
The composed deformable sleep/settling thresholds and settling damping are
authored zero, as are rigid sleep/stabilization thresholds. Scene TGS and
disabled stabilization are recorded. The public rigid sleep getter reads USD;
it is not independent native-actor verification.

A separate physics-only experiment removes the combined intermediate
display-update, zero-time rendering, annotation readback and image-writing
block, retaining initialization and the same physical-state getter timing.
Its q, v, handle positions, commands and complete trajectory archives are
bitwise identical to the captured original 32-iteration case for zero force,
the diagonal and reset replay. It produces **no images and no sensor-QC
certificates**. This intervention did not change that trajectory; startup
effects and other hidden mechanisms remain untested.

## Matching rigid iterations and verifying native material/rest: negative result

A four-cell TRAIN t0001/960 Hz audit matches both rigid handles' position
iteration counts to the deformable count, 32 or 64, in original and recentered
coordinates; rigid velocity count is 1. All cells pass full QC. Positions for
zero force, the diagonal and reset replay are bitwise identical to their
earlier same-condition rigid-16 baselines. Matching those counts therefore
does not fix the response. Translation disagreements remain 233.214 µm at 32
iterations and 662.680 µm at 64 iterations.

Native tensor getters verify material and rest geometry separately in all
three branches of all four cells: E = 4249.36767578125 Pa and
nu = 0.34071844816207886 equal the authored float32 values; rest points match
bitwise and the oriented tetrahedral multiset matches. Native rigid masses
and inertias are also recorded. No native deformable nodal mass/density getter
is available. These observations rule out a material/rest mismatch in those
12 checks; they do not establish the integration/coupling law's accuracy.
The rest check concerns the tensor-backend-managed rest-point buffer and
oriented topology. Pinned public upstream code initializes this buffer from
parsed rest positions; it does not read the internal cooked constitutive
inverse-tet matrices. Those matrices remain unverified, and equivalence of
that upstream code to the installed binary is unproved. E/nu are genuine
backend material-view getters, but no direct native body-to-material index
getter was verified. Thus the checks do not prove internal rest-matrix or
material-assignment correctness. This scope applies to the native checks in
the later diagnostics below.
The [matched-count summary](../examples/gaussian-neural-boundary/campaign-diagnosis/matched-rigid-native-state-v02.json)
retains native-state, validation, source/protocol and trajectory hashes.
The preliminary v01 runtime-verification failure remains preserved.

## Scene-PGS requests: coordinate agreement still fails

One completed TRAIN t0001/960 Hz original-coordinate cell requests scene PGS
with matched rigid/deformable position count 32. Full sensor/physics QC and
native material/rest checks pass for zero, diagonal and reset branches. The
diagonal coordinate RMS is 309.058 µm, and its difference from the matched
scene-TGS reference is 371.300 µm. Runtime minimum volume ratio is 0.993445
versus approximately 0.939887 in the earlier scene-TGS diagonal branch.
This changes the response and avoids the same compression level in this cell;
it does not demonstrate physical correctness or numerical convergence.

The authored scene option and successful execution do not reveal which
internal solver handled the deformable: no direct native solver-identity
getter is exposed. The
[PGS-request summary](../examples/gaussian-neural-boundary/campaign-diagnosis/original-pgs-request-v01.json)
preserves that limitation and includes metadata for the associated
`native-energy-proxy-pgs-v01.json`. Those elastic/kinetic quantities remain
independent declared continuum proxies, without a native energy-conservation
verdict. The subsequently completed translated 32-iteration cell also passes
full QC and native material/rest checks, but translation-undone diagonal
disagreement is 42.096 µm, above its predeclared 15.453 µm tolerance.

Two further predeclared cells repeat both coordinate conditions at matched
rigid/deformable position count 64. Both pass full QC and native material/rest
checks. Translation disagreement becomes 42.704 µm, above 15.445 µm tolerance
and **1.44% larger** than at 32 iterations. The 32→64 iteration differences
are only 3.966 µm in original coordinates and 3.669 µm after recentering, each
below its engineering tolerance. However, two levels provide only one
iteration difference per condition: they cannot establish contraction or
iteration convergence, and the coordinate gate still fails. See the
[completed PGS pair/iteration summary](../examples/gaussian-neural-boundary/campaign-diagnosis/scene-pgs-coordinate-iterations-v01.json).
Scene-PGS changes the response but does not qualify a numerical reference.

## Free rigid force-API controls and explicit clock adaptation

Four controls isolate one free rigid handle with the same authored mass and
inertia, at 120/960 Hz and with the external-forces-every-iteration option
false/true. There is no deformable, attachment, gravity, collision or damping,
and no sensor-dataset claim. All four pass the predeclared 100 ppm cumulative
momentum gate and zero-torque rotation gate:

| Rate | Force option | Maximum cumulative momentum error | Fitted step impulse gain |
|---|---|---:|---:|
| 120 Hz | False | 0.293 ppm | 1.00000015 |
| 120 Hz | True | 5.851 ppm | 1.00000364 |
| 960 Hz | False | 0.710 ppm | 1.00000054 |
| 960 Hz | True | 21.078 ppm | 0.99998755 |

The original analyses failed after completing captures because they compared
the accumulated clock to nominal double-precision dt. A separately saved
exact clock/reference adapter verifies measured native intervals against
float32-rounded nominal World/manager getters, retaining the measured
intervals for the impulse reference. It changes no simulation, force, mass,
hardware, momentum tolerance or rotation guard. All original failures remain
saved; **no simulation was rerun** for the adapter.

Fitted position-convention theta is approximately 1 for the false option and
0.516 for true (constant-force exact integration would give 0.5). Position
conventions are diagnostic, not an additional force-accuracy pass. The
[free-control summary](../examples/gaussian-neural-boundary/campaign-diagnosis/free-rigid-force-api-v01.json)
includes frozen capture/adapter sources, input/trajectory/native-state hashes
and original failure hashes. The isolated force API has no large multiplicative
gain error in these controls; rigid-deformable coupling remains unvalidated.
All prediction, material-calibration and active-policy gates remain unpassed.

## Free whole-body momentum controls: no trustworthy mass estimate

Three further predeclared 960 Hz physics-only cells retain the same mesh,
E/nu/rho, internal elasticity damping and both full rigid attachments, while
removing the fixture and explicitly overriding background linear damping
from 0.593159/s to zero. The volume actor retains its collision construction,
but self-collision is explicitly disabled and there are no external colliders,
rigid collision shapes or gravity. An earlier damping-confounded v01 protocol
was prepared and never launched; v02 records this intentional intervention.

Each cell first tests unforced uniform co-translation at (0.001, 0, 0) m/s
for every node and both handles. It then starts independently from rest for
the diagonal pulse. Native E/nu, rest points/oriented topology and rigid
mass/inertia/COM checks pass; all nodal fixture flags are disabled. Full q/v
is sampled at 31 times, with volume-weighted body COM velocity, both rigid
velocities and measured native clock intervals recorded at every step.

The predeclared co-translation tolerance is 1% for both proxy total momentum
and body COM velocity drift:

| Scene request / coordinates | Unforced proxy momentum drift | Co-translation gate | Driven scalar mass fit | Driven declared-mass momentum residual |
|---|---:|---|---:|---:|
| TGS / original | 3.028% | Failed | −4.797 kg | 103.04% of pulse impulse |
| TGS / recentered | 5.440% | Failed | 3.328 kg | 100.06% |
| PGS / original | 0.02998% | Passed | 1.7368 kg | 14.685% |

The declared body mass is 1.837443 kg. Its COM uses normalized rest-tet-volume
nodal lumping, an explicit uniform-density assumption. No native deformable
mass getter exists. The TGS fits are incompatible and cannot be interpreted
as density estimates. PGS improves the unforced control, but even its best
scalar mass fit leaves **13.610%** driven momentum residual. Passing the
unforced control alone therefore does not validate that fit or an effective
native density. Unknown effective mass distribution, attachment coupling,
kinematic implementation and integration remain possible contributors.

The TGS diagonal trajectories also differ by 220.241 µm coordinate RMSE after
undoing translation. These are mass/momentum proxies rather than an exact
native conservation test; they do not establish a specific backend defect.
All six branches remain positive-volume and finite, and failures of the
engineering controls are retained. The
[whole-body summary](../examples/gaussian-neural-boundary/campaign-diagnosis/free-wholebody-momentum-v02.json)
records the original inputs, intentional override and source/evidence hashes.
No images, sensor-QC claim, material correction or predictive promotion follows.

## Bare uniform gravity: bulk sensitivity without attachments or mass ambiguity

Six further predeclared physics-only controls remove **both handles, every
attachment and all supports**, and set both background and elasticity damping
to zero. Self-collision is disabled; native body gravity is enabled. The same
TRAIN geometry and E/nu/rho are retained. A native gravity pulse with peak
acceleration (0.01, 0.01, 0.005) m/s² acts uniformly on all nodes. Its native
acceleration readback and measured float32-clock intervals determine the
analytic held-acceleration velocity reference. This reference is independent
of density or nodal masses: every free node should translate identically.

All six cells remain finite and positive-volume, and pass native E/nu/rest
checks and raw dynamic-target checks (fourth component 1 for every node).
The velocity gate was declared before execution: maximum vector error below
the larger of 2 µm/s and 0.1% of peak expected speed.

| Scene request | Rate | Coordinates | Maximum velocity error / peak speed | Maximum spurious nodal displacement nonuniformity | Velocity gate |
|---|---:|---|---:|---:|---|
| TGS | 120 Hz | Original | 15.755% | 14.982 µm | Failed |
| TGS | 120 Hz | Recentered | 0.02769% | 0.340 µm | Passed |
| TGS | 960 Hz | Original | 17.405% | 287.400 µm | Failed |
| TGS | 960 Hz | Recentered | 2.451% | 86.642 µm | Failed |
| PGS | 960 Hz | Original | 0.09397% | 0.411 µm | Passed |
| PGS | 960 Hz | Recentered | 0.07479% | 1.513 µm | Passed |

Thus a bulk numerical response failure occurs in these scene-TGS
configurations even without rigid attachments, an applied-force interface or
an unknown mass reference. Increasing the recentered rate from 120 to 960 Hz
also loses agreement with the uniform-acceleration reference. The scene-PGS
bare passes help narrow the diagnosis, but do not fix its separately observed
fixed-support translation error or coupled/localized-force discrepancies.
The authored scene options do not expose the internal deformable solver's
identity. No native energy-conservation result is inferred. The
[bare-gravity summary](../examples/gaussian-neural-boundary/campaign-diagnosis/bare-uniform-gravity-v01.json)
records all six cells, thresholds, guards and source/input/evidence hashes;
these outputs contain no sensor data and do not open any predictive gate.

## External-force iteration option: negative result

A separate frozen four-cell TRAIN t0001 audit explicitly authors
`physxScene:enableExternalForcesEveryIteration` at 960 Hz. The original
coordinates, material, supports, mesh and full hardware are retained. The
false/32-iteration cell reproduces the earlier baseline positions, velocities
and force-step commands bit for bit for zero force, diagonal pulse and reset.

| Force option | Deformable iterations | Diagonal RMS | Status |
|---|---:|---:|---|
| False | 32 | 236.556 µm | Full independent QC passed; baseline replay matched |
| True | 32 | 226.415 µm | Full independent QC passed |
| True | 64 | 710.965 µm | Full independent QC passed |
| True | 128 | — | Compression/support guard failed at 1.2 s |

The false32→true32 trajectory difference is 191.492 µm coordinate RMSE,
despite similar scalar motion amplitudes. With the flag enabled, increasing
32→64 iterations changes the trajectory by 594.243 µm, followed by a failure
at 128 iterations. The failed cell remains in the ledger. This option alone
does **not** fix numerical reliability. Identical logged force impulses do
not establish native intra-iteration integration semantics or accuracy.

The audit also records native rigid tensor getters: each handle has mass
approximately 1 g, and inverse masses, inertias and centers of mass are
retained. These observations do not verify effective deformable nodal masses.
The public [flag summary](../examples/gaussian-neural-boundary/campaign-diagnosis/external-force-iteration-flag-v01.json)
records source, protocol, analysis and trajectory hashes without copying
absolute failure tracebacks.

## Low-rate refinement and translation: negative result

A further frozen six-cell audit tests 60, 120 and 240 Hz at 32 deformable
iterations, in both original and recentered coordinates. Every cell completed
and passed the full independent sensor/geometry/physics QC. Their unchanged
31 physical sample times, midpoint force commands, float32 clocks and reset
guards permit matched comparisons without selecting a favorable rate.

| Physics rate | Original diagonal RMS | Recentered diagonal RMS | Translation disagreement |
|---|---:|---:|---:|
| 60 Hz | 412.042 µm | 284.697 µm | 316.811 µm |
| 120 Hz | 225.208 µm | 232.515 µm | 137.563 µm |
| 240 Hz | 168.788 µm | 46.899 µm | 162.117 µm |

After undoing the world translation, every diagonal trajectory disagreement
exceeds the predeclared 2 µm tolerance. Original-coordinate refinement
differences are 328.325→251.262 µm, with final tolerance 8.439 µm. Recentered
differences increase 149.921→207.524 µm, with final tolerance 2.345 µm.
Both refinement conditions and the translation gate therefore fail. Full
capture QC does not establish numerical convergence. The earlier
480/960/1920 Hz failure remains evidence; the low-rate results do not repair
the native reference or justify selecting a rate from these responses.
The [low-rate summary](../examples/gaussian-neural-boundary/campaign-diagnosis/lowrate-numerical-v01.json)
retains source/configuration, validation and trajectory hashes.

An additional metadata-only public summary documents a
[continuum energy proxy audit](../examples/gaussian-neural-boundary/campaign-diagnosis/continuum-energy-proxy-v01.json)
on the original/recentered 960 Hz trajectories. The audit computes a declared
corotated continuum elastic-energy proxy, declared-density lumped body kinetic
energy, and held-force translational work from the logged handle positions.
These are not native PhysX total-energy buffers. Native deformable effective
masses, rigid rotational kinetic energy and internal constraint energies have
not been established; the elastic proxy's equivalence to the installed backend
is also unverified. No native energy-conservation violation or conservation
pass is inferred from these quantities.

A separate eight-case CPU audit adds the documented **fixed-corotated**
volumetric term, giving
`mu * ||F-R||² + lambda/2 * (det(F)-1)²`, while preserving the earlier common
corotated source and curves. It covers both coordinates at 32/64 matched
counts for scene TGS and PGS. Changing the formula alters peak elastic proxies
by −0.036%, −0.377%, −2.480% and +0.698% for the TGS cases, and approximately
−0.07% to −0.09% for PGS. This correction does not resolve the discrepancy.
The documented PhysX 5.8 formula is not verified to equal the installed Isaac
implementation, and these are still independent continuum proxies with
unverified native mass/constraint energy. It establishes neither native
energy-conservation failure nor a pass. See the
[fixed-corotated proxy summary](../examples/gaussian-neural-boundary/campaign-diagnosis/fixed-corotated-energy-proxy-v01.json).

## Independent linear FEM sanity reference

A separately reviewed CPU reference uses the same TRAIN t0001 mesh (455
nodes, 1,728 tetrahedra), authored material, fixed patch and complete two-handle
hardware. It assembles small-strain isotropic linear tetrahedral elasticity,
eliminates fixed degrees of freedom exactly and condenses each rigid attachment
using infinitesimal translation/rotation. Both authored 1 g handle masses and
their authored inertias are included. No penalty stiffness or diagonal repair
is added. The reduced stiffness is positive definite in this representation;
the static relative force-balance residual is 2.07e-14.

At the diagnostic peak force (0.01, 0.01, 0.005) N, static coordinate RMS is
129.636 µm. For the diagonal pulse, implicit midpoint dynamics give:

| Rate | Dynamic coordinate RMS | Difference from preceding rate |
|---|---:|---:|
| 240 Hz | 77.106 µm | — |
| 480 Hz | 77.102 µm | 16.816 nm |
| 960 Hz | 77.101 µm | 4.212 nm |

Differences are measured at the same 31 saved physical times. Their contraction
ratio is 0.25045 and the final difference is below the declared 0.771 µm
tolerance, supporting second-order temporal convergence of this **linear
discrete system**. At 960 Hz the maximum principal small strain at saved times
is 0.523%; the full-step strain Frobenius bound is 0.637%, with displacement
gradient Frobenius norm at most 0.01133.

This is an independent sanity reference, not a native Isaac fix or a
replacement for nonlinear deformation prediction. Linearized rotations and
geometry exclude finite-deformation effects. The declared viscous damping
omits native `elasticityDamping`; its dynamic response is therefore not an
exact PhysX target. Neither mesh convergence nor native effective deformable
mass has been established. The
[linear FEM summary](../examples/gaussian-neural-boundary/campaign-diagnosis/linear-fem-t0001-v01.json)
includes input/source hashes and these qualifications. Known-force G1,
physical calibration and active probing remain unpassed; reserved predictive
gates remain closed.

## Independent finite-strain static sanity reference

The current `nonlinear-fem-static-t0001-v02` diagnostic adds compressible
neo-Hookean finite-strain elasticity and exact finite rotations of both rigid
attachment patches on the same TRAIN t0001 mesh and fixed support. Seven
predeclared load scales, 0.001→1 of the constant (0.01, 0.01, 0.005) N
diagnostic force, reach the declared translation and physical world-torque
residual tolerances. An independent saved-state nodal-gradient audit also
checks both handles and total external force/moment balance at every scale.
The peak tangent is positive definite in the declared generalized-coordinate
representation, without stiffness repair.

At peak load the nonlinear equilibrium has 129.241 µm coordinate RMS versus
129.636 µm for linear statics. Their coordinate difference is 0.422603 µm,
or 0.326% of the linear response. Maximum Green principal strain is 0.3915%
and minimum volume ratio is 0.999046. Thus finite-strain corrections are small
in this specific independent static model. This does not explain, correct or
validate the native dynamic trajectories: the constitutive law is declared
separately, inertia/damping are absent from statics, and the unchanged mesh has
not been refined. Continuation from rest establishes a local equilibrium,
without proving global minimality or branch uniqueness.

The [v02 static summary](../examples/gaussian-neural-boundary/campaign-diagnosis/nonlinear-fem-static-t0001-v02.json)
includes source/input, saved-position and independent reaction-audit hashes.
The preliminary v01 result and its axis-angle torque metadata remain intact;
v02 reports physical world torques using the SO(3) Jacobian conversion and
provides the separately computed nodal force/moment audit. No native,
dynamic-prediction, material-calibration or active-policy gate is passed by
this static sanity check, and reserved predictive gates remain closed.

## Nonlinear approximation and missing hardware context

Eleven matched pulse controls (three axes plus eight off-axis controls at two
magnitudes) identify a full-rank nine-column quadratic force design with
condition number 19.68. A measured zero-force trajectory supplies its fixed
offset. The independent TRAIN diagonal is opened only after the linear and
quadratic coefficients and predictions have been frozen; it selects nothing.

| TRAIN layout | Force-induced diagonal RMS | Linear error | Quadratic error |
|---|---:|---:|---:|
| t0000, point pin | 23.242 µm | 17.055 µm | 10.304 µm |
| t0001, fixed patch | 236.556 µm | 178.608 µm | 123.677 µm |
| t0002, two patches | 53.625 µm | 40.369 µm | 24.084 µm |
| t0003, moving patch | 320.340 µm | 207.756 µm | 230.518 µm |

Added polynomial capacity helps three layouts but worsens the fourth and does
not resolve the reference problem. Maximum local principal-stretch deviations
are 1.9%, 10.0%, 3.0% and 21.5%, so small whole-body motion does not guarantee
small local strain. PhysX documents a rotationally invariant corotated
elasticity model; finite-deformation effects are plausible. These observations
cannot separate that mechanism from the demonstrated numerical sensitivity.
See [PhysX deformable volumes](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/DeformableVolume.html)
and the [deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html).

A geometry-only audit of 44 fitting TRAIN layouts finds 12 same-handle-index,
same-center groups with different nine-node attachment sets (26 handle
instances), with authored inertia ratios up to 4×. No naturally repeated pair
of both centers was found in that subset; these scenes do not demonstrate
natural duplicates of the entire physical vector.

A controlled t0000 pair changes only the second handle's patch to a valid,
disjoint nine-node patch observed in t0026. Both centers, full encoded 25D
vector, recipe, supports, material, force and camera remain identical. Both
variants pass full QC and have identical zero-force responses. Their diagonal
responses differ by 6.972 µm RMS, about 30% of the original 23.242 µm signal,
with 187.669 µm maximum nodal difference. Their motion amplitudes are nearly
identical, so amplitude alone misses this effect. This establishes a
response-relevant omitted hardware variable in one pair, not a cause of all
forward error or of within-layout nonlinearity.

A future common known-hardware context must include both handles' reference
anchor coordinates, attachment node IDs, corresponding reference local
offsets in the **same node order**, mass, authored inertia and initial
orientation. Use actual common reference geometry; if sorting IDs, permute
offsets together. Supply the same information to every baseline and policy.
Alternatively these quantities must remain explicit latent nuisance variables
beyond the existing 25D representation. A centers-only designed prior must not
be described as the exact Isaac conditional prior.

## Evidence, sources and tests

- `runs/gaussian-campaign-v01/numerical-timestep-v01/`: three rates, immutable
  original errors, source-derived validator, input lineage and analysis.
- `numerical-solver-coordinates-v02/`: six-cell protocols, composed attributes,
  warmup/capture guards, five validations and preserved compression failure;
  `failure-aware-analysis.json` includes the failed cell explicitly.
- `numerical-solver-crossed-v01/`: separately declared crossed-device replay.
- `numerical-capture-backaction-v01/`: physics-only records, exact source diff,
  no-images/no-sensor-QC qualification and bitwise comparisons.
- `numerical-external-forces-v01/`: four predeclared scene-option/iteration
  cells, exact baseline replay, native rigid property records and preserved
  128-iteration failure.
- `numerical-matched-rigid-v02/`: four matched-count cells, native E/nu/rest
  checks across 12 branches and bitwise old-baseline positions; v01 preserved.
- `numerical-pgs-attempt-v01/`, `native-energy-proxy-pgs-v01.json`: one
  original-coordinate scene-PGS request, full QC, changed trajectory and
  qualified continuum proxy metadata, without native solver-identity proof.
- `native-free-handle-v01/clock-analysis-v01/`: four free rigid momentum
  controls, explicit measured-clock adapter and preserved original errors.
- `numerical-lowrate-v01/`: six completed and independently verified cells,
  failed rate/translation gates and retained higher-rate negative evidence.
- `native-energy-proxy-v01.json`: independently declared continuum proxies
  and logged held-force translation work, without a native conservation claim.
- `linear-fem-t0001-v01/`: independent linear discrete reference, frozen
  source/input hashes, static equilibrium and temporal-refinement evidence.
- `nonlinear-fem-static-t0001-v02/`: independent finite-strain static local
  equilibria at seven load scales and separately audited nodal reactions;
  preliminary v01 remains preserved.
- `numerical-attachment-audit-v02.json`, `numerical-hardware-pair-v01/`,
  `numerical-force-polynomial-four-v01/`, `numerical-rich-axis-replay.json`:
  hardware ambiguity, paired effect, held-diagonal diagnostics and replay hashes.
- `reports/gaussian-campaign-numerical-v03/`: static scientific PNG/PDF and
  manifest; generated by `experiments/campaign_numerical_report.py`.

The native audit implementation lives in `experiments/campaign_numerical_*`;
the free-handle controls and clock adapter are `experiments/campaign_free_handle_*`;
the independent references are `experiments/campaign_linear_fem.py` and
`experiments/campaign_nonlinear_fem.py`, and the proxy audit is
`experiments/campaign_native_energy_audit.py`.
Frozen workers/configurations have additional provenance beside the unchanged
base source. Tests cover identifiable cross-force terms, measured moving-support
offsets, rigid-rotation strain invariance, reserved-role rejection, original
preservation, midpoint impulses and float32 clocks, reset-guard retention,
attachment overlap prevention, physical/camera translation equivariance,
clock-preserving state mutation rejection, and explicit removal of sensor
claims from the physics-only derivative:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_campaign_numerical_*.py' -v
.venv/bin/python -m pytest tests/test_campaign_linear_fem.py -q
.venv/bin/python -m pytest tests/test_campaign_nonlinear_fem.py -q
```

The next physical-calibration claim requires the separately predeclared
numerical gate across all four support types. Passing a single numerical
factorial, even if achieved later, would not establish mesh convergence or
real-world material accuracy.
