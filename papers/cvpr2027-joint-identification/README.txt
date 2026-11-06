Seeing Motion Is Not Identifying Mechanics
Joint Material–Interaction Inference with Marginalized Loads

This manuscript studies physical identification from sparse visible-surface
movement, with known geometry, camera projection and point correspondences.
It does not claim a completed raw-RGB-to-force-field system.

WHAT IS NEW
The unknowns are spatial material contrast, compliant support location/stiffness,
and a spatial load footprint. Two temporal force histories are integrated out
analytically, not fixed to a single fitted answer. Posterior predictions preserve
material/force dependence. Additional camera tracks and calibrated reset loads
are compared against a fixed predictive objective. A separate hierarchical
experiment also infers uncertainty in the force-prior amplitude scale.
FEM supplies the forward response; the contribution is the joint identification,
uncertainty and evidence-selection system and its controlled evaluation.

MAIN EVIDENCE
reports/joint-field-surface-v04/ — surface-track finite/continuous posterior,
320 cases, fixed/random/acquisition controls, dependence and visibility audits.
reports/joint-field-hierarchy-surface-v05/ — independent128-case force-scale
mixture, nominal/broader/oracle controls, separate four-view/four-probe branches.
reports/joint-field-temporal-v06/ — fresh64-case temporal-covariance stress follow-up;
full mixture versus amplitude-only/fixed/covariance-oracle controls.
reports/joint-field-fine-v03/ — separate32-case coarse/fine numerical reference,
interior material sensors, not a camera experiment.
reports/joint-field-identification-v01/ and joint-field-hierarchy-v02/ — historical
interior-sensor mathematical references. Preserved, not pooled with surface data.

All primary test cases are retained. No final case selects a schedule or prior.
Means of per-case coordinate RMSE differ from volume-weighted nodal visual errors.
Time indexes replayed quasistatic load states, not inertial future forecasting.
No gravity, impact, friction, unilateral contact or unknown geometry is inferred
by the new reference. These capabilities in earlier platform experiments use
separate adapters and are preserved in the archived supplement.

BUILD
Run bash build.sh from this directory. It uses Tectonic if available at the project
runs/toolchain/tectonic path or on PATH; otherwise latexmk. Poppler pdfunite appends
the unchanged historical supplement to the new study supplement. Source archive
is release/latex-source.zip. The source archive includes all figure PDFs and the
historical supplement; it does not require experiment banks to build the paper.
The anonymous submission identifier is intentionally unset until registration.

REPRODUCTION
See docs/JOINT_FIELD_IDENTIFICATION.md for protocol/commands and version scope.
The scripts under scripts/joint_field_identification are the implementation.
Published compact summaries, CSVs, frozen hashes, numerical audits and source
allow checking reported values. Raw banks/recordings/checkpoints stay local.
Do not rerun --freeze into a completed report directory: retain frozen evidence.
Use a separate checkout/output namespace for new experiments.
