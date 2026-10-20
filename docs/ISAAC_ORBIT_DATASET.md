# Orbiting-camera deformable-block dataset

The user requested several Isaac scenarios with varying material, forces,
support types and locations, plus a camera that moves around the object.
`isaac-boundary-orbit-v0.1` is a frozen 48-episode development dataset for that
request. This is forward simulation data for boundary learning; no inverse
boundary-recovery or uncertainty result is claimed.

## Scenario design

The full factorial design has three Young's moduli (1.5, 5 and 20 kPa), four
load families and four boundary setups. The block is 24 × 12 × 8 cm with
455 nodes, 1,728 tetrahedra and 4,096 material-attached Gaussian samples.
Every episode lasts 3 seconds, with a 960 Hz physics step and 61 RGB/depth/mask
frames at 20 fps. Forces use Newtons; all saved geometry uses metres.
Gravity is zero. There is no floor or external contact.

| Boundary | Constraint |
|---|---|
| Point pin | One fixed surface node; permits rigid rotation |
| Fixed patch | 16 fixed surface nodes |
| Two patches | Nine fixed nodes on each of two opposite faces |
| Moving patch | 16 nodes follow an 8 mm smooth tangential displacement and return |

Fixture face, patch centre, load faces/locations, force amplitudes and camera
path are seeded. Load patches never overlap fixtures or each other. The two
load handles each attach to nine nodes and add a known 1 g of mass. Patches
therefore have local rigidity; these are finite handle-applied loads, not
uniform pressure or ideal point tractions. The unused handle remains attached
and adds inertia even in a single-handle loading case.

| Force family | World-frame load history |
|---|---|
| Push pulse | Inward normal, squared-sine pulse, from 0.1T to 0.55T |
| Pull and hold | Outward normal, smooth rise/hold/fall from 0.1T to 0.9T |
| Oscillating shear | Tangent to the reference face, two tapered cycles |
| Twist couple | Equal and opposite forces on separated handles; zero net commanded force |

Directions are defined from reference geometry and remain fixed in world
coordinates. They are not follower loads. Nominal amplitudes are 0.12–0.24 N;
point-pin cases use one quarter of this range to limit large rigid rotation.
The complete schedule, handle application positions and resulting net moment
are saved at every physics step. Command minus handle mass times discrete
acceleration is saved as an inferred transferred force, not a measured contact
force. Fixture reaction forces are not recorded.

Poisson ratio varies over 0.28–0.42, density over 400–850 kg/m³, and linear drag
over 0.35–0.85 /s, independently of Young's modulus. Elasticity damping is 0.02.
These nuisance parameters, loads, support geometry and camera path repeat
across each matched stiffness triplet. Every material has the same blue/ochre
surface pattern; colour does not encode stiffness.

## Camera and observations

One actual Isaac RGB/depth camera makes a smooth 85–125° azimuth arc at roughly
0.9 m radius. Elevation also changes smoothly. The path looks at the rest
centre, independently of hidden motion or material. Each frame has its own
camera pose; the sensor resolution is 640 × 480, focal length 35 mm, aperture
36 × 27 mm. Matrices use USD row vectors, local −Z forward and +Y up. Pixel
coordinates have +v down; depth is positive camera-axis depth, background zero.
The intrinsics are fx=fy=622.222…, cx=320, cy=240. Pixel centres are (u+.5,v+.5).

Sparse inputs contain frame IDs `[0,5,10,15,20,40,45,50,55,60]`: ten observations,
with no input frames between t=1.00 and 2.00 s. Withheld frames permit temporal
completion evaluation. The moving single view leaves hidden surfaces at each
time; camera motion must not be mistaken for deformation.

`load_model_inputs()` in `experiments/isaac_dataset.py` reads only the reference
and sparse observations. Never pass the catalog, recipes, summaries, truth or
preview labels to a model. For the first boundary-only experiment, explicitly
supply the true material/density as a separate, declared known-material
condition. Joint material/boundary inference is a different evaluation.
Depth paths are available for sensor checks and explicitly declared RGB-D
experiments; ignore them for an RGB-only evaluation. Episode IDs and file names
are identifiers, never predictive features. The volumetric rest mesh is supplied
as known geometry in this first controlled setup.
The Gaussian reference is a geometry/appearance scaffold attached to surface
triangles, not an optimized 3DGS reconstruction. RGB is rendered from the
coloured mesh. A Gaussian renderer will have a model discrepancy to account for.

## Splits and files

There are 24 train, 12 validation and 12 test episodes. A group comprises one
load family, one boundary type and one seed, with three stiffness variants.
All members stay in one split. A Latin-square rule ensures each split contains
all four load and boundary types. This is a small controlled dataset with only
16 independently sampled boundary/load layouts, not evidence of broad shape
or real-world generalization. The plan is declared before inspecting outputs:
`configs/isaac_boundary_orbit_v01.json`.

```text
plan.json, catalog.json, file-manifest.json, logs/
episodes/e0000/
  scene.usda                  # initial scene; use the worker to replay loads
  reference.npz               # rest mesh, colours and Gaussian attachments
  observations/
    input_protocol.json, cameras.npz
    rgb/, depth/, masks/      # only 10 observed frames
  truth/
    recipe.json, boundaries.json
    trajectory.npz, cameras.npz
    rgb/, depth/, masks/      # all 61 frames
    visibility.npz, metrics.json
  provenance/                 # exact generation sources
  summary.json, validation.json
```

Dense trajectory data contains nodal positions/velocities, advected Gaussian
means, boundary targets, handle positions, all force-step commands/application
positions/moments, nodal masses and transmitted-force estimates. Camera NPZs
contain `K`, `camera_to_world_usd_row`, `times_s`, and `frame_ids`.

Truth visibility labels are `0=hidden`, `1=visible`, `2=ambiguous`. They use
front-facing material triangles and bilinear inverse rendered depth, with a
0.5 mm visible tolerance and >1.5 mm occlusion gap. Raster boundaries and
uncertain depth agreements remain ambiguous; exclude them from a strict
visible-versus-hidden score. These labels describe raster visibility, not
exact analytic visibility. Elastic coordinate RMS removes best-fit rigid
translation and rotation; motion about a pin does not count as deformation.

## Generate and verify

Use the installed Isaac Python without modifying its environment:

```sh
.venv/bin/python -m unittest discover -s tests -p test_isaac_dataset.py
.venv/bin/python experiments/run_isaac_dataset.py \
  --output runs/isaac-boundary-orbit-new --gpus 0 1 2 \
  --isaac-python '/path/to/isaac/python'
.venv/bin/python experiments/validate_isaac_dataset.py \
  runs/isaac-boundary-orbit-new/episodes/e0000
.venv/bin/python experiments/describe_isaac_dataset.py \
  runs/isaac-boundary-orbit-new --output runs/orbit-results.json
.venv/bin/python experiments/audit_isaac_dataset.py runs/isaac-boundary-orbit-new
.venv/bin/python experiments/preview_isaac_dataset.py \
  runs/isaac-boundary-orbit-new --output runs/orbit-preview
```

The launcher has bounded workers/timeouts, freezes sources and plan, and resumes
only verified successful episodes. Failed paths are preserved and are never
silently overwritten or replaced. A fresh output is needed for changed physics.
An additional GPU 3 startup returned invalid sensor capture shapes at time zero;
that attempt is preserved under `rejected-attempts/` with its log and source.
`auxiliary-execution.json` records the failed capture; the same frozen e0039
recipe is generated on the primary queue. CPU performs geometry,
force/sensor replay, visibility postprocessing and archive integrity checks.

Independent validation checks recipe/source hashes, mesh orientation, masses,
all force steps/moments, boundary targets, clocks, rigid load attachments,
Gaussian advection, camera paths/intrinsics, RGB/depth/mask counts and sparse
byte equality. It ray-traces 240 saved pixel centres per episode against the
saved nodal mesh and requires depth agreement within 0.2 mm. It also checks
framing and finite data. Runtime additionally checks every physics step for
constraint drift and element compression/inversion; minimum volume ratio must
remain at least 0.1.

These checks establish internal consistency at the declared discretization.
Spatial and temporal convergence and calibrated continuum mechanics are not
established. A point pin is mesh dependent. Large compression or unknown
contact requires additional physical validation. No learner has been trained,
no test-set performance measured, and Bird remains stopped.

See the [generated evidence, preview and raw archive](../examples/isaac-boundary-orbit/README.md).
