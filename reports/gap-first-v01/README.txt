GAP-FIRST PILOT: REPRODUCTION AND EVIDENCE GUIDE

Question: when does uncertainty about a physical explanation change the
outcome of the next specified interaction, and what additional evidence helps?

The user added the bowl after the original campaign objective was recorded.
The campaign therefore contains three studies. They are distinct experiments,
not a single newly trained raw-video-to-physics network:

1. DROP — released ReconPhys appearance, mechanics, renderer and predictor.
   Scaling mass, spring stiffness and damping produces nearly identical drops
   but different responses to the same known body force. A separate calibration
   trial supplies a rendered image-centroid observation. The two-candidate
   posterior and an equally informed MAP estimator achieve the same corrected
   point accuracy. A separately frozen extension scores the actual unchanged
   network's prediction; it does not pretend that calibration retrained it.

2. STEM — independently verified beam with uncertain bending/base compliance.
   Initial tip displacement identifies their sum. The next task may be safe
   without further evidence, resolved by a visible-base observation, or require
   a known-force probe when permitted cameras cannot reveal the base. A third
   load location is withheld from calibration. The strong Gaussian moment-risk
   baseline performs similarly to full posterior risk in this reference study.

3. BOWL — nonlinear soft solid in a concave rigid bowl with gravity, contact,
   prescribed-position pushing and bowl rotation. Calibration and held-out
   pushing use different trajectories. Motion and reaction-force predictions
   are separate quantities. A finer-space/time simulation is a numerical-model
   stress reference, not certified continuum truth. The secondary resolution
   mixture explicitly includes that finer model by construction.

READING ORDER
  REPORT.txt                         completed campaign narrative/results
  completion-requirements.json       release audit requirements and evidence
  stem/summary.json                  all stem policies/conditions
  stem/independent-outcome-audit.json independent numerical/dataflow audit
  drop/README.txt                    native reproduction and attribution
  drop/independent-scientific-final-review.json
  drop/native-reference-extension/summary.json
  bowl/README.txt                    nonlinear study reproduction
  bowl/numerical-qualification.json  retained numerical limitations/failures
  bowl/summary.json                  matched and discretization-stress results

Publication is gated on the complete bowl results, audits, final PDF inspection
and standalone source rebuild. A requirements file alone is not proof of
completion; inspect its final status and the underlying evidence.

SAFE REPRODUCTION
Do not start duplicate simulations in the campaign's original running workspace.
Use a fresh checkout or a separate directory. Existing scientific outputs are
protected by source/bank/prediction hashes and no-overwrite checks.

The stem needs Python, NumPy and SciPy. From the repository, using its environment:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.gap_first_v01.stem_reproduce \
    --workspace /tmp/gap-first-stem-reproduction --workers 4
The destination must not already exist. This copies the required sources into
the new workspace, seals predictions, evaluates outcomes in a separate stage,
and compares the complete summary with the published reference. It has been
run successfully in a fresh workspace on the campaign machine.

For the drop, follow drop/README.txt. Download upstream assets and weights
directly; this repository does not redistribute raw datasets, checkpoints or
vendor source trees. Native replay uses its separate Python 3.10 / Torch 2.5
environment; the stem/bowl environment is different. The wider illustration
camera is not the frozen quantitative camera and has no real-view PSNR claim.

For the bowl, follow bowl/README.txt and the frozen protocol. Three resumable
workers can share GPUs 0–2, with a CPU-only finalizer. GPU 3 is excluded. Future
reproductions require a newly recorded wall-clock deadline/protocol; do not
silently edit the original protocol or overwrite its outputs. Keep the exact
mechanical parameters, inputs, priors, sensor noise, targets and tolerances when
reproducing the scientific experiment.

INTERPRETING SCORES
Stem errors are scalar displacement errors in millimetres. Drop targets are
21-coordinate centroid RMS in normalized native scene units, not SI material
measurements. Bowl motion is point-Euclidean RMS in millimetres; secondary
reaction forces are vector RMS in newtons after a declared 20 ms trailing boxcar.
Do not pool these quantities as one physical-accuracy score. Different image
PSNR aggregation/mask protocols are explicitly labeled in the drop reports.

Noise repeats and paired task/resolution variants are not additional independent
objects. The stem has 96 independent physical cases; the drop has 3 asset
geometries; the bowl has 16 material/contact worlds on one nonbox geometry.
All reported uncertainty is conditional on the declared hypotheses, priors,
observation models and numerical solvers. Failed adapter qualification is not
relabeled as native-framework failure. Corrected point accuracy alone does not
establish a superior acquisition policy.

ARTIFACTS
Videos and data-driven figures: output/gap-first-v01/{drop,stem,bowl}/.
Paper addendum: papers/cvpr2027-joint-identification/gap-first-addendum.pdf.
Editable standalone source: release/gap-first-latex-source.zip under that paper.
The earlier main.pdf, supplement.pdf and original source release are preserved.

Large report JSON files may be distributed as deterministic .json.gz files.
Decompress a copy outside the frozen report directory for inspection. Raw
simulation arrays, upstream videos, assets and weights remain local. The
publication manifest records compressed and decompressed hashes where relevant.

RELEASE GENERATION (AFTER ALL SCIENTIFIC WORKERS FINISH)
  python3 scripts/gap_first_v01/campaign_report.py
  python3 scripts/gap_first_v01/package_addendum.py
These standard-library-only release tools require the bowl finalizer's terminal
results-audits-and-figures-complete status, matching summary/outcome/prediction
hashes, all 576 bowl records and 96 audited trajectories, and complete stem/drop
cohorts. They refuse partial results, even if a partial summary file exists.
They do not require the later publication audit, avoiding a circular dependency.
Bad scientific outcomes do not fail this completeness gate: they remain in the
report. Numerical nonfinite/missing records require review, not silent exclusion.

The source ZIP includes only explicit LaTeX/style dependencies and referenced
PDF/PNG/JPEG figures inside the campaign output directory. It contains no raw
simulation archives, checkpoints, vendor trees, videos or user attachments.
Unexpected TeX/bibliography dependencies must be explicitly handled before
packaging. Archive creation is not a successful-build or visual-layout claim.
Extract the archive into a fresh directory and run `sh build.sh`; Tectonic
needs its standard TeX bundle (network on first use, or a populated local cache).
This standalone document build does not depend on Torch, CUDA, simulation
outputs outside the archive, or the original workstation filesystem.

The final publication audit separately verifies source hashes, independent
scientific audits, PDF compilation/appearance and the release manifest. The
requirements inventory is not itself that terminal audit. Preserve earlier
releases; publish the addendum as an addition with its own checksums.
