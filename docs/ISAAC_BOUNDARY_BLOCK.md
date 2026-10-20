# Isaac Sim boundary-learning starting scene

The new [48-scenario orbit dataset](ISAAC_ORBIT_DATASET.md) adds varied materials,
load/support setups, moving cameras, grouped splits and sparse inputs. The
single-scene description below records the earlier starting experiment.

**Revised flexible video:** the initial20kPa/single-pin example mostly rotates.
Use `bash experiments/launch_flexible_block.sh --output runs/flexible-new --keep-open`
for the1kPa, fixed-patch, finite-load-patch demo. See the
[matched material comparison and results](../examples/isaac-flexible-block/README.md).
The original configuration below remains available as an explicitly separate
experiment. The new9-node load patches are attached to low-mass rigid handles;
their geometry and inertia are recorded rather than treated as point forces.

Bird was stopped at the user's request on20 October2026. This experiment
starts the boundary-condition direction with a forward simulator, before
implementing boundary inference.

The separately developed [Gaussian boundary beam benchmark](GAUSSIAN_BOUNDARY_FORMULATION.md)
provides an analytic inverse reference with supplied2D correspondences. This
Isaac scene adds volumetric3D simulation and RGB data; the two pipelines have
not yet been coupled.

## Run

The launcher uses the existing Isaac Sim6.2 environment in the neighboring
`deformable worldmodel` project without modifying it. Override `ISAAC_PYTHON`
with another Isaac Python executable when needed. Choose a fresh output path.

```sh
# Visible Isaac simulation; pause the completed scene for inspection.
bash experiments/launch_isaac_block.sh --output runs/block-new --keep-open

# Different random pin and loads, or a four-node fixture variant.
bash experiments/launch_isaac_block.sh --headless --seed 31002 --output runs/block-other
bash experiments/launch_isaac_block.sh --headless --fixture-nodes 4 --output runs/block-patch

# Zero-load control; --no-rgb saves physics data without recording images.
bash experiments/launch_isaac_block.sh --headless --force-scale 0 --no-rgb --output runs/block-zero
.venv/bin/python experiments/validate_isaac_block.py runs/block-new
.venv/bin/python experiments/validate_isaac_block.py runs/block-zero --unloaded
.venv/bin/python experiments/plot_isaac_block.py runs/block-new --output runs/block-preview
```

The initial Isaac startup can take about a minute. `--keep-open` pauses the
completed simulation for GUI inspection; it does not implement interactive
user manipulation yet. The script, rather than opening the USD alone, applies
the recorded force schedule. Each command gets exactly one direct physics step;
the actual simulation clock is asserted at every step.

## Physics and boundaries

- A24×12×8cm elastic block:455 nodes,1,728 positive tetrahedra,12×6×4 cells.
  Cell counts are configurable. Known Young's modulus20kPa, Poisson ratio0.35,
  density500kg/m³, elasticity damping0.02 and explicit linear drag0.5/s; block mass1.152kg.
  Deformable and handle sleeping are disabled, as is artificial settling damping.
- One seeded random surface node fixed at its rest location by a vertex-to-
  kinematic-anchor attachment. A single pin permits rigid rotation. The
  optional nearest-node patch constrains more motion and is recorded explicitly.
- Three seeded surface load nodes, at least55mm from the fixture, with random
  world directions, amplitudes and smooth squared-sine pulses. Default peak
  forces are0.125–0.5N; `--force-scale` controls amplitude.
- Newton-valued forces act at each step on dynamic1g handles, each attached to
  one load node. Handles have no colliders and add known inertia. Loading node
  positions remain dynamic. The stored transferred-force estimate is command
  minus handle mass times discrete acceleration; it is not a contact sensor.
- Gravity is zero and there is no floor, isolating fixtures and applied loads.
  General contact, compliant supports and boundary switching come later.

The implementation uses the installed PhysX volumetric deformable system,
following NVIDIA's [deformable and attachment documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_bodies.html).
It runs in Isaac6.2/OmniPhysics110.4, rather than a legacy4.x soft-body API.

## Recorded evidence

| File | Contents |
|---|---|
| `scene.usda` | Self-contained initial Isaac scene |
| `reference.npz` | Rest mesh and4,096 material-attached Gaussian reference samples |
| `observations/rgb-*.png` | Actual640×480 Isaac renders,20fps over3s |
| `observations/camera.json` | Intrinsics, camera pose and conventions |
| `truth/boundaries.json` | Pin/load node IDs and pulse parameters |
| `truth/trajectory.npz` | Nodal motion, advected Gaussians, handle motion and all force-step commands |
| `provenance/` | Exact source copies checked against saved hashes |
| `summary.json`, `validation.json` | Runtime settings and independent integrity replay |

The Gaussians are a geometry/appearance scaffold with triangle barycentric
attachments, not an optimized3DGS reconstruction. Their scales describe
appearance, not posterior uncertainty. Current RGB comes from Isaac's colored
surface renderer; a future Gaussian likelihood must account for this difference.
Do not expose `truth/`, truth-bearing summaries or diagnostic plots to a learner.
Dense single-camera recordings are development data; sparse schedules,
visibility labels and held-out evaluation cases still need a frozen protocol.

Three numerical tests check mesh volume/mass/topology, load impulse in Newton-
seconds and Gaussian attachment. Actual GPU checks include finite states,
fixture drift, positive volumes, attachment consistency and force replay.
Unloaded, second-seed, patch, stiffness and time-step variants provide physical
checks. The [results and actual RGB preview](../examples/isaac-boundary-block)
report their outcomes. Elastic motion is measured after best rigid alignment,
so rotation about a point pin is not counted as deformation.

Point supports/loads are mesh dependent near their application points. A
smaller time step alone does not establish spatial convergence. No boundary
recovery, continuum accuracy or uncertainty calibration is claimed. Next:
freeze material/observations, infer hidden boundary hypotheses, and evaluate
boundary recovery and hidden motion against the saved truth. Use finite
patches and mesh refinement before interpreting boundary stresses.

The final explicit-step240/480Hz check still has timestep sensitivity: peak
elastic coordinate RMS0.717/0.674mm and peak total displacement231/183mm
(the latter includes rotation). Keep this a fixed-discretization development
simulator; numerical convergence is not established. Earlier app-clock and
default-sleeping trials are archived as development only.
