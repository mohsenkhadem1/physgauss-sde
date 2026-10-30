Coupled rigid/deformable cross-interaction campaign v13
=====================================================

Frozen before validation/reserved generation. Protocol, source hashes and
numerical development qualification are in reports/coupled-contact-v13.
Maximum campaign time is configs/coupled_contact_v13.json; its stage field is
initial setup metadata. Live progress is runs/coupled-contact-v13/stages.
Prior v11/v12 source and evidence are preserved by prior-preservation.json.

Post-freeze spatial audit
-------------------------
The original vertex-contact reference FAILED spatial consistency. In development,
4-to-6-cell edge-drop trajectories differ by 12.13 and 29.72 mm even though
timestep, volume and penetration checks pass. Area weighting alone does not
resolve a narrow support footprint. Original results remain comparisons against
that frozen simulator and must not be presented as validated physical truth.
See mesh-development-diagnostic.json and physics-final-audit.txt.

A separate surface-contact repair is under development in surface_*.py and
reports/coupled-surface-v13. It integrates fixed material surface patches and
distributes forces through the actual tetrahedral barycentric interpolation;
equal/opposite rigid forces and torques are retained. The old source, runs and
test cases are untouched. New geometry/contact compliance and eight fresh
parameter combinations are declared in protocol-design.json. Time, mesh AND
contact-quadrature consistency must be qualified before the new source freeze
and reserved generation. A design file alone is not a completed experiment.

Physics and task
----------------
Compressible tetrahedral Neo-Hookean FEM, objective strain-rate damping,
lumped SI masses and explicit symplectic integration. Surface nodal area
weights keep penalty-contact stiffness comparable under mesh refinement.
Moving spheres exchange equal/opposite force and torque with the soft body;
static planes, boxes and spheres define the tools/environment. Coulomb-capped
friction is dissipative. Positive element volumes are checked every step.
This is node-sampled penalty contact, not continuous collision detection or
self-contact. Dynamic nonspherical rigid bodies are not supported.

Observe one squeeze/shear/release trajectory at 18 tracked surface nodes with
0.1mm Gaussian position noise. Fit unknown stiffness and friction by a fixed
hypothesis grid. Predict four NEW interactions: sliding, edge dropping, curved
obstacle pushing, and pushing into a freely moving sphere. Initial shape,
density, gravity, scene geometry and future tool trajectories are known. Future
object trajectories and material/friction labels do not enter calibration.
The oracle diagnostic receives true parameters explicitly and is not a learned
competitor. Every new interaction starts from its supplied initial state; this
is transfer of inferred parameters, not continuation of the calibration state.

Reference uses a finer mesh and smaller timestep than the predictor. Both share
the constitutive family: this is controlled synthetic transfer, not independent
physical validation or a revised public-dataset benchmark result. Eight reserved
objects have different parameter combinations, within the search range; the
OOD claim concerns interaction changes, not unseen material-range extrapolation.

Comparisons
-----------
FEM: 7 stiffness x5 friction candidates, selected only by calibration MSE.
Native spring core: authors' unchanged PhysTwin spring-force kernel, with shared
v13 mesh graph, contact, SI masses and integrator; own 7 effective-scale x5
friction grid. This is NOT full PhysTwin, its native optimizer or its collision
system. Upstream PhysTwin supports multi-object collision; no incapability
claim is made. Effective spring scale is not Young's modulus.
Rigid+contact: identical geometry/contact, rigid mass and inertia,5frictionfits.
Position ridge, position rigid and persistence are calibration-only empirical
baselines; they lack scene geometry/contact and are not the strongest controls.
All method failures and grid-edge selections remain explicit. The shared positive-volume gate also rejects native spring trajectories that collapse inherited tetrahedra; this is a study-specific constraint, not a property of the full authors' method.

Qualification and reproducibility
---------------------------------
Development three-level timestep checks require decreasing full-trajectory error
and <=0.25mm medium-to-fine RMS; contact penetration <=1mm; informative shape
change after removing rigid motion. A failed curved configuration is archived.
Each actual reference episode has its own half-dt consistency check with finite
state/time, matching sample grids, volume and penetration gates on BOTH levels.
Qualified-only summaries require both calibration and forecast references to
pass. All declared cases remain in accounting, including unqualified references.
Two levels establish an absolute consistency check, not a convergence order.

Use runs/phystwin-env/bin/python. Required installed packages: numpy, scipy,
pytest, matplotlib, imageio-ffmpeg, torch and warp (native core only). Upstream
PhysTwin checkout stays local under runs/third_party/PhysTwin; kernel provenance
is verified against its Git HEAD. No raw data/checkpoints are published.

Entrypoints
-----------
qualification.py --workers 4 reproduces development numerical qualification.
pipeline.py library --workers 4 generates the fixed calibration hypotheses.
pipeline.py reference --split development|validation|reserved --workers 4
pipeline.py evaluate --split development|validation|reserved --workers 4
run_campaign.py executes these stages durably with per-stage completion markers.
campaign.py --launch NAME COMMAND starts a systemd user service bounded by the
absolute campaign deadline. Workers check the deadline and source freeze.
report.py summarizes coverage, errors and paired comparisons by split/scenario.
render.py renders three prespecified reserved cases, common world axes, 4x slower
playback explicitly labeled. It never substitutes more favorable examples.
A future reproduction needs a NEW deadline/output root and a newly declared
freeze; do not modify this frozen campaign's sources or reuse stale job caches.
