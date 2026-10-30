Surface-contact and consistent-mass extension (development)

This work preserves the original v13 files and all archived v11/v12 evidence.
New production outputs belong under runs/coupled-surface-v13 and published
artifacts under reports/coupled-surface-v13. Development diagnostics also have
uniquely versioned directories under runs/coupled-contact-v13. Raw arrays,
external assets and native checkpoints stay local.

Numerical method
clipped_surface_contact.cpp/.py integrates normal contact exactly over clipped
P1 rest-surface triangles for planes and boxes. Friction uses separately
refined quadrature; spheres use conservative sparse material-surface sampling.
This is penalty contact, without CCD or self-contact. The compiled cache records
source/compiler/flags/binary hashes. If setting CXX, use its absolute executable
path. The installed compiler fallback is recorded in generated provenance.

consistent_mass.py is a separate P1 consistent-inertia candidate. It factors
M_ff once, applies three Cartesian solves per explicit step, uses Mv for
momentum, and integrates rigid inertia from continuum tetrahedral volume.
Contact forces/damping retain their declared common law. FEM and the native
spring-force adaptation must share the same mass/integration/contact settings.
Static fixed-node constraints are supported; prescribed moving constraints
require an explicit different formulation. This does not add multiple soft
bodies, hand geometry, calibrated parameter uncertainty or active probing.

Evidence and selection
protocol-design.json and protocol-pending.json are prospective declarations,
not authorization to generate reserved cases. physics-development-qualification
must pass for the chosen numerical settings. method-freeze.json and protocol.json
are created only after all source aliases and comparators are audited. Mesh
pair differences demonstrate two-resolution consistency, not a rigorous bound
on continuum error. Failed development trials remain part of the evidence.

The prospective extension has two development and eight reserved material/
friction combinations, one squeeze/shear/release calibration and four forecasts.
Initial state, geometry, gravity, density, Poisson ratio, viscosity and future
tool positions are known. Unknown stiffness/friction are selected using observed
calibration positions. No future object targets select hypotheses. Interaction
families have already been used in numerical development. The native comparator
uses the authors' spring-force kernel in a matched numerical framework; it is
not the full official PhysTwin method.

Every calibration/forecast reference has a base run, a half-timestep run and a
spatial companion from the scene's preselected mesh pair: 150 reference runs
for ten objects and five interactions. Primary and companion use the same base
timestep; both mesh members must pass development time checks. Spatial metrics
use 27 common material points and 2678 surface probes, plus collider centers.
Failures mark references unqualified; they never remove cases or change the
primary target. The independent verifier uses a separate closed-form Cartesian
tetrahedral interpolator. Mesh/timestep agreement is a numerical consistency
check, not a proof of continuum truth or empirical material validity.

Execution after freeze
Use runs/phystwin-env/bin/python. surface_run_campaign.py --workers 12 runs
library, development references/evaluation, then reserved references/evaluation.
surface_finish.py --workers 12 waits for completion, runs matched supplements,
reporting, independent verification and predetermined videos. These commands
retain the original absolute campaign deadline; they do not start a new budget.
The isolated surface pipeline reuses original code in a private module namespace
and must bind all scene/contact/mass functions to the selected extension.

Recovery and reproduction
Machine-readable state: runs/coupled-contact-v13/campaign-state.json.
Service prefix: coupled-contact-v13-. Do not recreate live work or overwrite a
frozen source. This campaign ends no later than 2026-10-31 05:01:31 UTC. A later
reproduction needs a separate checkout/output namespace, a new explicitly
recorded deadline and new source/protocol hashes; never edit a published freeze
in place. Actual final mesh/timestep choices are in the frozen protocol, not
in this development README.

Archived benchmark diagnostic
benchmark_transfer is a separate 21-case development-only study of volume-based
COM/inertia in the previous reduced rigid-pose release model. It replays the
original predictions exactly before changing the released suffix. All held and
observed-prefix states remain unchanged; PhysTwin copies are unchanged because
that pipeline does not call this release model. It is not a test of the coupled
elastic solver. The small mixed gain was independently verified and has not
been promoted into the archived benchmark method.
