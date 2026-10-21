# A conditional physical prior for a matched numerical reference

Implemented in [campaign_physical_prior.py](../experiments/campaign_physical_prior.py), with [pure geometry/density tests](../tests/test_campaign_physical_prior.py). This module defines a **new designed conditional prior**. It does not condition the old learned joint flow and does not claim to reproduce the conditional distribution of the seeded Isaac generator. No numerical posterior, new trained inverse model, calibrated likelihood, or successful inference result is established merely by implementing this prior.

The inputs are public metric reference points and both public hardware handle centers. An optional `handle_patch_node_ids[2,9]` argument supplies the complete attachment geometry as explicit extra public information; it is never inferred from a center. No recipe, motion response, image, checkpoint, private boundary label, or reserved gate/calibration/confirmation outcome is read. The default reference is the existing cuboid; the class derives its bounds from the supplied points. Cameras are part of a later observation model, not an additional material constraint here.

## Random variables and measure

The continuous state has twelve coordinates:

| Coordinates | Distribution | Role |
|---|---|---|
| log E | Uniform(log1500, log20000), Pa before log | Young modulus |
| log rho | Density exp(log rho)/450 on log[400,850] | Physical density uniform400..850kg/m³ |
| nu | Uniform(.28,.42) | Poisson ratio |
| log damping | Density exp(log damping)/.5 on log[.35,.85] | Physical damping uniform.35...85s⁻¹ |
| support0 local(u,v) | Independent Uniform(0,1) | Position within a selected certified cell |
| support1 local(u,v) | Independent Uniform(0,1) | Second patch, or proper inactive nuisance |
| base force amplitude | Uniform(.12,.24)N | Multiplied by.25 for a point pin, as in the generator |
| couple angle | Uniform[-pi,pi) | Direction perpendicular to hardware-center separation; otherwise nuisance |
| moving direction angle | Uniform[-pi,pi) | Direction tangent to the selected support face; otherwise nuisance |
| moving amplitude | Uniform(.006,.010)m | Deliberately uncertain around the generator's fixed.008m; otherwise nuisance |

The material marginals match the continuous fresh simulator design, including physical-uniform density and damping rather than log-uniform substitutes. The moving-amplitude interval is a new modeling choice, not a learned empirical distribution or a claim that the original generator randomized this amplitude.

Discrete coordinates are boundary type(4), passive load family(4), support face(up to6), two support-cell indices(up to64 each), and an incident face for each known hardware center. Boundary and load types are independent uniform categorical variables. Each feasible support face is uniform, each eligible cell on that face is uniform, and local cell coordinates are uniform. Consequently this is uniform over eligible cells within a face, **not** surface-area-uniform across differently sized faces. In center-only mode an edge/corner hardware center has multiple incident face categories with equal mass; no arbitrary normal is silently selected. In full-hardware mode the incident faces must contain all nine patch nodes, normally identifying the original load face uniquely. The second hardware face remains a normalized nuisance when the selected force model does not use it.

For a state z with categories c and public geometry/hardware H,

```
p(z,c | H) = p(BC) p(load-family) p(support-face | BC,H)
              p(cell0 | BC,face,H) p(cell1 | BC,face,H)
              p(load-face0 | H) p(load-face1 | H)
              p(logE,logrho,nu,logdamping)
              p(localUV0) p(localUV1) p(A) p(couple-angle)
              p(moving-angle) p(moving-amplitude).
```

`log_prob` includes all categorical probabilities and the rho/damping Jacobians. Its measure is twelve-dimensional chart Lebesgue measure times counting measure on the discrete categories. The map to the existing25D physical vector repeats or drops inactive variables, has fixed known hardware coordinates, and is noninvertible. **There is no25D physical-space Lebesgue density.** Inactive variables are retained as proper finite distributions; discarding them and retaining their old density factors would define a different model.

## Support feasibility with explicit normalization

The generator chooses support centers within normalized face tangents[-.18,.18]² and selects the nearest1(point pin),9(two patches), or16(fixed/moving patch) face nodes. The prior partitions this central rectangle into a deterministic8×8 atlas. A cell is retained only when every possible center in it is certified to give a compact support patch whose selected nodes are more than.027m from **both public handle centers**, or from **all18 attachment nodes** when the complete public patches are supplied.

Let c be the metric cell center, r its metric half-diagonal, and Rk(c) the distance to the kth-nearest reference node on that face. For any x in the cell, the kth-nearest-node distance at x is at most Rk(c)+r, and each selected node is within Rk(c)+2r of c. The implementation enumerates the finite superset of potentially selected face nodes satisfying `|node-c| <= Rk(c)+2r`. It measures each such node against all supplied public hardware points and requires every distance to exceed.027m. This is tighter than replacing the node set with a solid bounding sphere, while preserving a certificate for every center in the cell. Rk(c)+r must also be at most the generator's.09m compact-patch radius. Two-patch supports use opposite faces and require eligible cells on both. Other boundary types decode the second support to the first; their second cell is a uniform nuisance over all atlas cells. If a BC has no feasible face, construction raises an error instead of silently deleting that BC's prior mass.

All retained cells and probabilities are explicit. There is no stochastic rejection loop, estimated acceptance rate, or omitted truncation normalizer. The certificate is conservative: some simulator-valid supports can lie in rejected cells. The resulting posterior is conditional on this designed support restriction, not the original generator posterior. Changing grid size, clearance, public hardware, or motion bounds changes the prior fingerprint and requires both numerical and amortized methods to use that new prior.

## Passive and controlled decoding

`decode(state)` returns `physical[N,25]`, `boundary_type[N]`, and `force_type[N]` for the existing neural forward interface. The first four values are logE,logrho,nu,logdamping. Both load centers equal the public conditioning inputs from the start. Push and pull are inward/outward reference normals; shear follows the existing deterministic face tangent. A twist is an equal/opposite pair with direction uniformly distributed in the plane perpendicular to handle separation. Moving supports have an unknown tangent direction and an unknown amplitude. The existing3s waveform family is supplied as a category; force and support timing uncertainty is not introduced in this version.

For a known reset action, `decode(state, known_control=menu_action)` replaces both old force vectors and the old waveform family. The deployed single-load interface uses handle0, accepts an arbitrary known world `peak_force_N` vector and `force_type_index`, and preserves the second hardware center, material, support geometry and prescribed motion. A known twist produces an equal/opposite pair. Changed contact centers, handle1 single-load commands, delayed pulses or another duration are rejected; those need a separately specified forward/control interface. These are reset experiments, not counterfactual rewinds inside an ongoing trajectory.

Original passive load variables remain in the joint state. They affect passive-video likelihoods; they are unused by a purely controlled-response likelihood and then integrate to one under their normalized conditional prior. This is a legitimate nuisance marginalization. Overwriting samples from the old learned joint flow would not produce this conditional prior.

## Hardware completeness is a separate issue

The original25D representation supplies handle centers, not full rigid-body attachment geometry. The Isaac generator chooses each nine-node patch using a jittered target and the already available surface nodes; it then records the nearest node as its center. A center alone does not reconstruct the full patch, its rigid-body local offsets, attachment lever arms, or inertia/orientation convention. In center-only mode the support atlas certifies separation from public centers, **not** separation from every node of an unknown actual handle patch. With `handle_patch_node_ids` it certifies separation from all actual public attachment nodes. Its force-couple directions use center separation, which is also an approximation to exact hardware mechanics.

Accordingly, this prior is currently complete for the **declared25D kernel surrogate**, not a complete latent description of the original Isaac mechanism. A numerical posterior can match that surrogate and observation likelihood exactly in model specification while both remain mechanically misspecified. It must not be called the true simulator posterior.

Two defensible extensions are:

1. Declare the complete reference attachment patches, rigid local offsets and mass/inertia convention public hardware descriptors. Supply the same descriptor to every prior, forward model, inference method and policy. Record that extra side information and its hash.
2. If those descriptors are unknown, introduce explicit attachment-geometry nuisance variables, a normalized conditional distribution, and a forward model that takes them. Marginalize them jointly with BC/material/load variables.

The first extension is partially represented explicitly: `hardware_descriptor()` supplies the2×9 reference node IDs/positions/local offsets, both1g masses, the worker's authored diagonal-inertia approximation with its3mm floor, and initial identity orientation. Full-patch geometry also conditions the prior's feasibility and load-face probabilities. These descriptors are **not** silently inserted into the25D vector or consumed by an existing forward model; that forward/context integration remains necessary. The second extension is not implemented here. The TRAIN-only hardware ambiguity audit can establish whether center-only conditioning is sufficient empirically; a good density calculation cannot repair omitted forward inputs.


## TRAIN-only support coverage audit

Coverage here means that each **actual TRAIN support center** falls in an eligible atlas cell under the declared boundary type. It is a prior-support check, not a posterior-coverage result. The108 audited TRAIN layouts exclude old calibration cases t0012..t0015 and all new calibration cases t0112..t0127. No response, reserved gate, validation, TEST or fresh-confirmation geometry was used.

The initial solid-sphere certificate covered only54/108 layouts(50%) and was rejected as too restrictive. Its source-pinned negative audit remains in `runs/operator-dev/campaign-prior-train-support-audit-v01.json`. The tighter finite-node certificate improves center-only8×8 coverage to79/108. Adding all known attachment nodes imposes the physically stricter constraint and gives:

| Atlas grid per face | Full-hardware TRAIN centers covered | Excluded layouts |
|---|---:|---:|
|8×8|67/108|41|
|16×16|87/108|21|
|32×32|97/108|11|
|64×64|101/108|7|
|128×128|105/108|3|
|256×256|107/108|1|

At128×128, excluded cases are t0021(fixed patch), t0035(moving patch), and t0041(fixed patch). At256×256 only t0035 remains excluded. The finer-grid check evaluates exactly the same certificate at each actual containing cell without constructing or sampling a huge full atlas. Detailed cases and certificates are retained in `campaign-prior-train-support-audit-v02.json` and `campaign-prior-train-support-refinement-v01.json` under `runs/operator-dev/`.

The constructor deliberately retains its cheap explicit8×8 default; it has **not** been adopted as a population-calibrated physical prior. A restricted numerical pilot must declare its grid, report excluded support cases without outcome-based replacement, and compare only to an amortized model using exactly that same prior. Even105/108 TRAIN inclusion at128×128 is not proof of full-population support. A future adaptive atlas or exact order-k Voronoi partition could reduce conservative exclusions without introducing a hidden rejection normalizer.

Local(u,v) moves stay inside one selected cell. The generic SMC implementation currently changes cells through joint-prior independence moves, which can mix poorly under a concentrated likelihood; fine grids make this especially important. Structured cell transitions/global support-position moves need their exact proposal ratios, and convergence requires independent-run/mode/particle-count checks. Neither final uniform weights nor a high local-MH acceptance rate establishes category or support-location mixing.

## A genuinely matched numerical/amortized comparison

After a forward model and working observation likelihood are frozen, both methods must target

```
pi(z,c | y,H) proportional to p(z,c | H) L(y | forward(decode(z,c,H)),H).
```

For combined passive and reset-probe data, use the same joint state and the product of their declared conditional likelihoods, with shared discrepancy/correlation represented explicitly rather than assuming independence without qualification. The precise likelihood must include the same public cameras, visibility handling, covariance, units, action waveform and hardware descriptor. It must not use hidden BC/material labels to choose noise.

A numerical reference may start from this prior and use sequential Monte Carlo with incremental weights `L**(beta_next-beta_old)`, resampling, valid continuous proposals and discrete BC/face/cell/load-family moves. Metropolis steps require the prior ratio **and proposal ratio**, especially where eligible face/cell counts differ. Report independent-run agreement, effective sample sizes, category occupancy and evidence/trajectory-score stability. A very small weight ESS or stuck BC occupancy is not a reference solution. There is no exact Gaussian load marginalization through nonlinear mechanics or censored rendering implied by the prior; use explicit nuisance integration/particles unless a separately stated linear model justifies an analytic integral.

An amortized posterior must be trained or otherwise validly adapted to this same designed joint prior and same likelihood, with known hardware in its inference context. Compare its marginal material **and BC/support** distributions and posterior predictive scores against the converged numerical reference. Three normalized density points, a fixed-BC stiffness slice, or hardware coordinates overwritten after old-flow sampling cannot establish that match. A stiffness-only pilot can diagnose a subproblem but cannot satisfy the full campaign objective.

Minimal CPU usage:

```python
prior = ConditionalPhysicalPrior(public_reference_points, public_handle_centers)
state = prior.sample(1024, np.random.default_rng(42))
log_prior = prior.log_prob(state)
passive_parameters = prior.decode(state)
known_parameters = prior.decode(state, known_control=public_action)
resampled = state.take(ancestor_indices)
# Torch continuous gradients, retaining the exact same discrete state:
coordinates = torch.tensor(state.coordinates, requires_grad=True)
parameters = prior.decode_torch(state, coordinates)
log_prior_torch = prior.log_prob_torch(state, coordinates)
```

No live models, reserved responses, new simulation, GPU jobs or publication changes were needed to define or test this prior.
