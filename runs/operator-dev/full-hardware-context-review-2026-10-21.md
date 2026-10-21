# Complete hardware/query interface

The new `experiments/campaign_full_hardware_context.py` defines an explicit, structured input contract for a future forward model. It does not modify or wrap an existing learned model. The old 25D operator and sealed checkpoints remain unchanged; prediction, inference and calibration have not improved merely by adding this interface.

The old `gaussian_operator.py` coefficient schema stores material scalars, support/load centers, resultant forces and one moving-support vector. Those centers cannot identify the complete rigid attachment footprint or determine the handle's mass, full COM inertia, COM offset and initial orientation. The new interface preserves these inputs instead of compressing them into centers.

## Typed inputs and information roles

All mesh arrays, hardware properties, commands and hypotheses must be supplied explicitly. There is no episode/recipe/image/trajectory loader, path-based label extraction, hidden reference-mesh generator or fallback material.

| Group | Contents | Public image-conditioning input? |
|---|---|---|
| Known hardware | Rest nodes, tetrahedra, complete surface connectivity, complete ordered rigid patches, node masks, origins, masses, full COM tensors, COM offsets and initial rotations | Yes |
| Query times | Exact force interval boundaries and requested output times | Yes |
| Known force programs | Explicitly tagged controlled handle forces, pure torques, nodal loads and uniform body acceleration | Yes |
| Known support program | Prescribed endpoint motion explicitly tagged as a known actuator command | Yes |
| Material hypothesis | Homogeneous E, Poisson ratio, density, body drag and constitutive-family hypothesis | No |
| Boundary hypothesis | Full node-axis support masks and support origins | No |
| Latent force programs | Sampled/diagnostic passive-force histories, separately tagged as hypotheses | No |
| Latent support law | Unknown prescribed support motion, separately tagged as a BC hypothesis | No |

`inference_public_context(known_hardware=..., command_query=...)` returns only the public feature groups and their public provenance. It cannot accept material, boundary, RGB, response or truth arguments. Candidate force/support values and their source fingerprints are excluded from this public result. Merely having the true passive-force history or moving-support amplitude in a recipe does not make it a known command.

`conditioned_forward_context(...)` additionally requires explicit material and boundary objects, a declared hypothesis role (`candidate_from_prior_or_posterior` or `known_parameter_training_or_diagnostic`), and source records. It adds clearly separated hypothesis groups. This supports both matching-prior forward queries and privileged known-parameter training without placing those labels into public image conditioning.

Forcing is represented by additive `ForceProgram` objects. Each must explicitly declare `known_actuator_command` or `candidate_force_hypothesis`. Thus a known poke can coexist with an unknown passive-force history. `combined_forward_forcing(...)` sums the supplied programs only for an explicitly conditioned physics/forward call; its result is marked unsuitable for public image conditioning. Support programs similarly require a known-command or candidate-BC role.

The interface enforces declared separation. It cannot determine whether a caller falsely tagged an unobserved recipe value as a known command. Source hashes and provenance are audit metadata, not encoder features.

## Mechanical completeness and coordinate convention

Every handle retains the complete ordered patch IDs, an H-by-N node mask, and padded ordered body-local offsets with an explicit validity mask. The same load center with a different footprint therefore produces a different representation. Patch order is preserved even when its membership set is unchanged. Ragged patch sizes and zero-handle free bodies are supported.

All point features are relative to a caller-supplied world origin, with world axes retained. For initial rotation R0, the body-local patch offsets are `(rest_patch_world - handle_origin_world) @ R0`; reconstruction uses the inverse map. The full COM inertia is expressed in that body frame, and the COM offset is a separate vector. Derived world inertia, COM position and inertia about the handle origin are exposed without discarding off-diagonal terms. Handle forces act at the moving handle origin; additional pure torques are world vectors. Handle mass is additional to deformable body mass.

The existing independently tested `campaign_rigid_inertia.py` helper checks symmetric positive inertia and physical triangle inequalities. No diagonal-only approximation or stiffness/inertia repair is introduced. Exact finite rigid attachments are part of this contract; native attachment compliance or equivalence is not implied.

The material hypothesis is kept separate from public geometry and hardware. In particular, no oracle density-derived body mass is added to public features. Uniform body acceleration is an explicit command, so known gravity need not be turned into density-labelled nodal forces.

Boundary hypotheses retain full support membership per node and per axis, not only centers. The support law supplies exact discrete endpoint SO(3) transforms and translations about hypothesized support origins. Candidate programs may use inactive, all-false masks; the number of program channels is not the observed active-support count. No continuous interpolation is silently inferred.

## Schema, provenance and operating scope

`schema()` exposes each named feature's symbolic dimensions, units, frame and information role. The public representation is structured; there is no automatic flattening into a lossy 25D vector. Node, element, face, handle, patch-padding, support-channel, force-program and time dimensions remain explicit.

Inputs are copied into canonical finite arrays. Complete mesh boundary faces, nondegenerate tetrahedra, proper rotations, physical inertia, index validity and unambiguous hard attachments are checked. Feature-return arrays are copies. Provenance records array shape, dtype and exact byte SHA256; source fingerprints cover this interface and the inertia helper. Caller-supplied source records are explicitly labelled as declarations that were not opened or independently verified. Absolute-frame input hashes are retained as provenance/output-restoration metadata while the model features use relative coordinates.

Relative features are translation-consistent to floating-point roundoff; exact provenance hashes need not be translation-invariant because supplied absolute bytes and subtraction roundoff can differ. The demo's maximum feature difference after a large global translation is 5.55e-17.

Version 0.1 describes an initially undeformed object at t=0 with zero velocities and supplied initial rigid orientations. Initial support commands must agree with that rest state. Requested output times must have explicit support endpoints on the command grid. Arbitrary preload, initial motion, continuous support interpolation, contact environments, compliant attachments, heterogeneous constitutive fields and soft/Robin BCs require an explicitly extended contract. This preparatory interface does not claim to solve generalized BC uncertainty or cross-shape generalization.

## Evidence

34 targeted tests pass in 0.24 s using at most two CPU threads: 26 interface tests plus eight existing rigid-inertia tests. They use a literal five-node mesh and explicit commands, not simulator episodes. They demonstrate:

- Different rigid patches sharing the same load origin remain distinguishable; ordered membership and ragged padding are preserved.
- Off-diagonal inertia and COM offsets change physical kinetic energy; an independent point-mass calculation agrees with the full-tensor expression.
- All model features and derived support targets remain consistent under global translation.
- Changing material, latent force/support values and latent source hashes leaves public conditioning unchanged, while explicit known actuator commands remain available.
- A known probe and a latent passive history remain separate until an explicitly conditioned forward sum.
- Candidate material/BC/force/support groups remain separate, public APIs reject RGB/response/truth arguments, and nested provenance hashes cover their complete child records.
- No dataset/archive is implicitly opened; only this interface and the inertia helper's source bytes are read for provenance.

`runs/operator-dev/full-hardware-context-demo-v01.json` contains a literal interface example, full known-hardware features, group descriptors, units/schema and source provenance. It is not a deformation prediction, simulated validation or model-quality result.

No training, architecture change, GPU job, simulation, reserved response access or checkpoint modification occurred. The three inspected old forward source files have no local diff; their hashes are recorded in the accompanying inventory.
