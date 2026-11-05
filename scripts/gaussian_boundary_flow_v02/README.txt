Gaussian boundary inference: audited development campaign (v02)

Purpose
Learn distributions of invisible mechanical boundary parameters from an observed
motion prefix, then predict motion under a supplied future command. Geometry,
cameras, action trajectories and an initialized attachment prior are currently
inputs. This is not yet contact discovery from arbitrary RGB video. Probability
flow time is distinct from the mechanical simulation clock.

Preserved evidence
The v01 scripts/results and previous Isaac, SoftVT and PhysTwin comparisons remain
unchanged. New candidates have separate files/run directories. Do not replace old
best results with a newer candidate merely because it has a richer architecture.

Controlled inference
matched_campaign.py compares capacity-matched flow, Gaussian-mixture and point
regressors on clean simulated worlds with one acquisition-noise draw. The diagonal
mixture improves predictive means; flow remains a joint-uncertainty comparator.
full_covariance_campaign.py records the unsuccessful extra-capacity ablation.
reference_posterior.py validates only a small two-parameter CONDITIONAL posterior.
differential_action.py tests informative shape change after removing rigid motion.

External development pipeline
1. benchmark_inputs.py exports only prefix observations, initial geometry and
   supplied actions. Evaluation targets have separate loaders. Fresh SoftVT
   trajectories are guarded until a new protocol/checkpoint is frozen.
2. export_inputs.py persists pickle-free packages with byte and semantic hashes.
   Always reuse the same package: triangulation across SciPy environments can
   change the observation operator.
3. benchmark_readiness.py checks initial geometry and all intermediate solves.
4. package_campaign.py builds a geometry/command-specific simulation bank, trains
   flow and mixture inference, seals their samples, and then computes forecasts.
   --geometry is explicit: the dense-surface filtering rule is inappropriate for
   sparse volume nodes and can reject all useful SoftVT tetrahedra.
5. object_score.py / softvt_score.py check seals before opening held suffix targets.
   Raw mechanical states and rebased tracking trajectories are separate channels.
   Joint samples are transported before quantiles; node marginal intervals cannot
   be mapped into pointwise intervals. Persistence/velocity get the same selected
   observations. Solver rejection and out-of-bank observations remain visible.

Example (shared GPU 0; raw artifacts stay local)
.venv/bin/python -m scripts.gaussian_boundary_flow_v02.package_campaign \
  --geometry convex \
  --package runs/gaussian-boundary-flow-v02-inputs/object-soft-task1-demo0 \
  --output runs/gaussian-boundary-flow-v02-package/object-soft-task1-demo0 \
  --samples 128 --steps 5000 --draws 32 --device cuda:0
runs/phystwin-env/bin/python -m scripts.gaussian_boundary_flow_v02.softvt_score \
  --run runs/gaussian-boundary-flow-v02-package/object-soft-task1-demo0

Physics qualification
object_prestrain.py adds objective natural strain; boundary_preload.py adds localized
boundary reactions. These are operators, not demonstrated video identification.
finite_range_attachment.py gives a conservative Gaussian cohesive grasp: weak
bonds can release under gravity with a fixed active hand. It is not Coulomb contact.
robust_dynamic_step.py and release_safeguarded_v3_qualification.py qualify two
bounded no-impact release cases. Impact convergence remains unqualified; the old
simulation evidence is not retroactively replaced.

Reports
reports/gaussian-boundary-flow-v02 contains independent numerical/scientific/data
reviews, decisions.json, compact scores, failures, and the dataset inventory.
PokeFlex is unavailable without approved recordings. Deform360 remains paused.
All 22 local PhysTwin sequences are historically exposed. SoftVT has 600 local
episodes but only three assets; new episode tests are not unseen-material tests.
Do not describe a development pilot as a sealed all-dataset benchmark.

Observation-budget clarification
SoftVT uses 12 inverse-fitting frames plus a prefix grasp-frame pose calibration
from the same 48 sparse points. In the four development cases this is 13 distinct
object frames. Mechanics and retrieval share this construction; simple velocity
and persistence use the 12 selected frames and do not exploit future tool commands.
See grasp-initialization-budget-audit.json. This disclosure changes no predictions.
