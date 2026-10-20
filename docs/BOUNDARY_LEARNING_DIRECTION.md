# Research direction: learning boundary conditions

Decision record from the user's supplied discussion, reviewed 20 October 2026.
This records the research direction; the benchmark below is a proposed
implementation plan, not an implemented or completed experiment.

## Primary question

Given a static 3D Gaussian reference and sparse visual observations, can we
infer uncertain supports, contacts and boundary loads well enough to recover
hidden deformation, then improve those beliefs with new views or measured
physical interactions?

The main focus is **boundary-condition learning**. This includes where a
condition applies, its type, parameters and activation history, rather than
only fitting a force magnitude. Start with known geometry, cameras and
material. Introduce material uncertainty after testing boundary recovery.
Spatial completion and temporal interpolation are the initial motion tasks;
forecasting needs a separately declared assumption about future interactions.

## What the representation must distinguish

| Boundary condition | Unknown quantities |
|---|---|
| Fixed or prescribed displacement | Patch, constrained directions, anchor and motion history |
| Applied traction | Patch, direction and spatial/temporal coefficients |
| Free surface | Whether zero traction is appropriate |
| Compliant attachment | Patch, anchor, support stiffness and damping |
| Contact | Obstacle, activation, sticking/sliding and friction |

Use persistent material surface patches, including hidden patches. An image
silhouette is not the mechanical boundary. Keep permanent material connections
separate from temporary contact relations. A fixture's reaction force follows
from its displacement constraint; do not fit an independent arbitrary force
on the same constrained degree of freedom.

Retain a distribution over plausible boundary configurations when observations
cannot distinguish them. Unknown support compliance, material stiffness and
load amplitude can compensate for one another; an optimizer returning one
configuration does not establish identifiability.

## Architecture proposed in the discussion

1. Associate masks, features and tracks with persistent objects in the static
   Gaussian reference. A frozen DINO encoder and existing video segmenter are
   candidate perception components, to be introduced after controlled mechanics.
2. Attach many appearance Gaussians to fewer mechanical nodes. Preserve material
   identities; update Gaussian geometry consistently with deformation. Choose
   solid, shell or rod mechanics to match the object.
3. Propose boundary patch types, parameters and persistence, with uncertainty.
   Test proposals against mechanics and the observed images. Begin with a small
   set of connected candidate patches, rather than a force on every Gaussian.
4. Update motion frequently, boundary/contact beliefs as evidence changes, and
   material beliefs across informative observations. A sliding-window smoother
   is a candidate for sequential inference; offline sampling audits uncertainty.
5. Distinguish a virtual displacement edit from a measured physical action.
   Only observed real responses provide new material/boundary evidence. A
   prescribed displacement alone may not identify absolute stiffness; measured
   forces or informative dynamics with known mass may be needed.

Conditional Gaussian traction coefficients can use the project's exact load
marginalization when the residual is affine in those coefficients. The result
must retain covariance normalizers. Integrating coefficients does not integrate
unknown fixture location/type or nonlinear contact activation. Those hypotheses
need explicit inference or a validated approximation.

## First benchmark: proposed scope and acceptance evidence

Start with one synthetic elastic beam or block, known rest geometry, material,
mass and cameras, and a complete static Gaussian reference. Supply masks and
associations initially. Hide a support or loaded patch from the input camera.
Vary support location/compliance and load location/history independently; begin
with a finite candidate-patch set and simple elastic mechanics.

Compare under exactly the same observations:

- An oracle with the true boundary conditions.
- Mechanics with a fixed, potentially incorrect boundary assumption.
- Geometric deformation fitting without inferred mechanical boundaries.
- The proposed uncertain-boundary model.

Report boundary type/location recovery and parameter intervals where
identifiable, motion error separately on observed and hidden regions, image
error, physical constraint residuals, and trajectory uncertainty coverage and
proper scores. Evaluate temporal gaps separately from any later forecasting
experiment. Use multiple independently generated cases for calibration; one
successful trajectory is insufficient. Declare train/development/test splits,
visibility labels, seeds, observation schedules and acceptance thresholds before
inspecting test outputs. Synthetic truth should come from an independently
checked solver with spatial/time refinement.

Then reveal another view or apply a known real/synthetic physical interaction
and test whether boundary ambiguity and hidden-motion error decrease. Keep
unidentifiable examples: appropriate uncertainty is an outcome, not a failure
to be removed by choosing an easier point estimate.

## Relationship to the existing implementation

The current code supplies mechanics, Gaussian rendering, structured load
marginalization and posterior audit infrastructure. Its declared load profiles,
supports and floor treatment do **not** implement general unknown boundary
patch/type inference. Bird's reduced modal model and fixed floor are an
inference test, not the proposed boundary-identification benchmark.

The already authorized local CPU/GPU Bird run continues unchanged through its
declared 2,000/4,000/8,000-draw gates. Preserve its evidence and frozen source;
do not reinterpret it as boundary-learning evidence. The wider historical
comparison still requires convergence and independent uncertainty readiness.
Boundary-model changes require separate targets, protocols and experiments.

The separate `bird_collapsed_state.py` prototype tests Gaussian velocity/load
elimination for the constant-noise case. Its density/derivative/geometry tests
pass, but no transition recovery, production mixing or convergence result has
been established. It is not integrated into the active worker.

The papers and SOTA claims mentioned in the pasted discussion have not been
independently verified for this decision record. Obtain exact primary-source
links and check their settings before choosing literature baselines or making
novelty claims. No new literature result is asserted here.
