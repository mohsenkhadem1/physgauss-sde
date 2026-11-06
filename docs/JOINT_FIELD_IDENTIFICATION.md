# Joint material–interaction identification

The new system estimates a **distribution of physical explanations of motion**.
Known-geometry FEM is the forward model, rather than the proposed contribution.
Spatial material, support and load fields remain separate; Gaussian temporal load
amplitudes are analytically marginalized conditional on those fields. A posterior
can then predict response to a new known load, report uncertainty, and compare
additional camera tracks with calibrated reset probes.

## Pipeline

1. Supply metric geometry, calibrated orthographic cameras and known surface
   correspondences. Read sparse projected displacements with stated measurement
   noise. The evaluated system does not extract these from raw RGB automatically.
2. Represent positive heterogeneous elasticity with two normalized Gaussian
   material features in reference coordinates. Represent compliant support and
   external traction using separately normalized Gaussian surface footprints.
3. For each spatial hypothesis, solve the mechanical response to unit load modes.
   This constructs a linear map from temporal force amplitudes to observations.
4. Integrate those amplitudes with a correlated Gaussian prior. Retain the
   likelihood determinant and conditional force covariance. Weight spatial
   hypotheses by the marginalized likelihood.
5. Combine conditional predictions using total expectation and variance. The
   mixture retains dependence between material/support and unknown forces.
6. Compare expected reduction of a fixed predictive loss from another surface
   observation or a known reset load. Acquire only that measurement and update.

The finite dictionary has 3,125 spatial hypotheses; it is an exact *finite*
reference. Continuous SMC infers the same five coordinates with continuous priors,
while still integrating force amplitudes analytically. These priors differ at
endpoints; comparing them is not a pure quadrature convergence experiment.
The continuous study replays actions chosen by the finite planner.

The hierarchical extension uses five latent force-prior scale values, making
15,625 joint spatial/scale hypotheses. A separately frozen temporal follow-up
adds three force-correlation lengths, giving 46,875 spatial/covariance hypotheses.
The amplitude study compares with fixed nominal, variance-matched broader Gaussian
and true-scale oracle controls. The temporal follow-up compares with amplitude-only,
fixed nominal and true-covariance oracle controls. It tests a recognized temporal-prior
failure on fresh cases; it is not an untouched primary evaluation or prior-averaged
calibration test.

## Identifiability

Scaling elasticity, compliant support stiffness and unknown loads together leaves
unknown-load displacements unchanged. No camera resolves that conditional
forward-map ambiguity. A fixed force prior can prefer a scale; that is prior
information. A calibrated input does not co-scale and can distinguish responses.
This does not imply that four probes uniquely identify every material field.

## Evidence versions

| Version | Purpose | Sensors |
|---|---|---|
| surface-v04 | Primary spatial inference and acquisition, 320 cases | Visible front/back surface markers with valid camera projections |
| hierarchy-surface-v05 | Fresh force-scale uncertainty study, 128 cases | Same surface/projection contract |
| temporal-v06 | Fresh64-case conditional temporal-prior stress follow-up | Same visible-surface contract |
| fine-v03 | Coarse/fine inverse comparison, 32 cases | Interior material coordinates; mathematical numerical reference |
| identification-v01 | Original finite/continuous reference | Interior material coordinates |
| hierarchy-v02 | Original force-scale reference | Interior material coordinates |

The v04/v05 correction moved y coordinates from ±25 mm to the actual ±30 mm
surfaces and supplied image-plane projections and camera directions. Both received
fresh development/test seeds. Every deformed candidate observation/probe state is
ray-audited. The historical versions are retained and never relabeled as RGB
surface tracking. Visibility checking does not establish correspondence accuracy.

Primary targets contain equal numbers of hidden original-motion coordinates and
new known-load response coordinates. Original target points are never acquired;
target reset loads differ from acquisition loads. Four scalar measurements are
compared with strong development-selected fixed, camera-only, probe-only and
random policies. Measurements share the same case-specific noise across policies.

## Run and inspect

Run Python modules from the repository root with the project environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_joint_field_mechanics.py tests/test_joint_field_inference.py \
  tests/test_joint_field_continuous.py tests/test_joint_field_campaign_audit.py
.venv/bin/python -m scripts.joint_field_identification.surface_report
.venv/bin/python -m scripts.joint_field_identification.surface_evidence_audit
.venv/bin/python -m scripts.joint_field_identification.hierarchy_surface --self-test
```

The frozen protocols list source/bank hashes, seeds, bounds, counts and deadlines.
`surface_freeze.py`, `surface_campaign.py`, `surface_supervisor.py` and
`hierarchy_surface.py` contain generation/run entry points; inspect `--help` before
starting an independent replication. Completed-source hashes must not be edited
in place to resume a changed model. Raw banks and checkpoints stay local.

`paper_results.py` refuses incomplete cohorts in final mode; its explicit `--draft` mode marks the manuscript and omits unfinished continuous-cohort conclusions. It generates LaTeX results from
final surface/hierarchy summaries and records source hashes. `surface_figures.py`
and `surface_visualize.py` create scientific figures from actual saved results.
They do not use generative image editing. Replay videos show true-scale correlated
quasistatic states, not dynamically integrated future motion.

## Relationship to earlier work

The earlier sparse-RGB cube implementation in `src/physgauss/marginal_forces.py`
and `marginal_inference.py` integrated loads with a fixed spatial profile. The new
study makes spatial material, support and load hypotheses uncertain and compares
informative evidence. The earlier beam prototype in `docs/BEAM_ACTIVE_DESIGN.md`
already evaluated camera-versus-reset-probe selection with marginalized forces.
The new system extends that structure to joint 3D material/support/load fields
and a shared original-motion/changed-load prediction target. Conditional Gaussian elimination is established mathematics,
not a claimed invention. Related latent-force, Bayesian identification, Gaussian
interaction and active physical-twin work is cited in the manuscript.

Historical contact, SoftVT, real PhysTwin and PIDG studies remain in
`papers/cvpr2027-physical-response/` and the new archived supplement. They use
separate adapters. Their rendering/motion performance must not be attributed to
the new joint posterior, and public video does not supply spatial force ground
truth for the new recovery metrics.

## Independent replication without the local banks

The paper source ZIP is a **LaTeX build package**, not the experiment environment.
It includes every figure PDF and the archived supplement. Building it requires
Tectonic (or TeX Live with `latexmk`) and Poppler's `pdfunite`; no simulation bank
or Arial installation is needed to build these already-embedded PDF figures.
Regenerating figures from Python additionally requires Arial at the font paths
used in the figure scripts. The scientific calculation does not require that font.

The final repository release includes deterministic `all-case-metrics.csv.gz`
files for the finite, fine, hierarchy and temporal cohorts. `release_index.py`
exports the hierarchy/temporal rows from the complete case records. These compact
files allow recalculating means and paired case-bootstrap comparisons without raw
banks, images or posterior checkpoints. See the accompanying protocol for cohort
IDs, bootstrap seeds and geometry stratification. The CSVs are metrics, not inputs
for rerunning inference. Some figure/audit scripts need regenerated case records.

For an independent CPU replication, **use a new checkout**, never the directory
containing the frozen publication runs. The example below reproduces the primary
surface experiment. It preserves the published protocol in a separate evidence
directory, regenerates the unit-response banks, reconstructs the published fixed
schedules, and records a new protocol identity and deadline before any new cases.
A new bank may differ bytewise across CPU/GPU and numerical-library versions; its
hash belongs to this replication, not to the published frozen experiment.

```bash
git clone https://github.com/mohsenkhadem1/physgauss-sde.git physgauss-replication
cd physgauss-replication
python3 -m venv .venv
.venv/bin/pip install 'numpy>=2,<3' 'scipy>=1.15,<2' 'torch>=2.2' \
  'matplotlib>=3.8' 'Pillow>=10' pytest
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_joint_field_mechanics.py \
  tests/test_joint_field_inference.py tests/test_joint_field_continuous.py \
  tests/test_joint_field_campaign_audit.py

# This preparation is ONLY for the new checkout, before generating cases.
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
source = Path('reports/joint-field-surface-v04')
archive = Path('published-evidence/joint-field-surface-v04')
assert source.is_dir() and not archive.exists()
assert not Path('runs/joint-field-surface-v04').exists(), 'Use a clean checkout'
archive.parent.mkdir(exist_ok=True)
shutil.move(str(source), str(archive))
source.mkdir(parents=True)
PY

for shape in block step; do
  .venv/bin/python -m scripts.joint_field_identification.surface \
    --shape "$shape" --device cpu --batch 4 \
    --output "runs/joint-field-surface-v04/$shape.npz"
done

.venv/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json
archive = Path('published-evidence/joint-field-surface-v04/protocol.json')
report = Path('reports/joint-field-surface-v04')
p = json.loads(archive.read_text())
# Verify source identity rather than silently accepting changed algorithms.
for name, digest in p['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
for shape in p['shapes']:
    dest = report / shape
    dest.mkdir()
    (dest / 'fixed-schedules.json').write_text(
        json.dumps(p['fixed_schedules'][shape], indent=2))
    bank = Path(f'runs/joint-field-surface-v04/{shape}.npz')
    p['bank_sha256'][shape] = hashlib.sha256(bank.read_bytes()).hexdigest()
now = datetime.now(timezone.utc)
p['replication_of_protocol_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
p['frozen_utc'] = now.isoformat()
p['deadline_utc'] = (now + timedelta(days=7)).isoformat()
p['replication_note'] = 'Independent CPU replication; published evidence retained separately'
(report / 'protocol.json').write_text(json.dumps(p, indent=2))
PY

for shape in block step; do
  .venv/bin/python -m scripts.joint_field_identification.surface_campaign \
    --shape "$shape" --phase test --kind matched --cases 64 --device cpu
  for population in continuous-coarse offgrid-fine force-prior; do
    .venv/bin/python -m scripts.joint_field_identification.surface_campaign \
      --shape "$shape" --phase test --kind "$population" --cases 32 --device cpu
  done
done
.venv/bin/python -m scripts.joint_field_identification.surface_report
```

This executes the **finite** estimator on all four truth populations; the
`continuous-coarse` name specifies continuous truth, not SMC inference. SMC is a
separate replay (`surface_supervisor.py` and its frozen continuous manifest).
The example retains the original fixed schedules for a direct replication.
Re-selecting them requires the separate development phase and a new declaration.

For the hierarchy and temporal follow-ups, retain the regenerated surface banks,
archive their published report directories in this new checkout, create empty
report directories, then use `hierarchy_surface --freeze` or
`temporal_hierarchy --freeze`, followed by `--shape block` and `--shape step`.
Before any temporal case is generated, replace the expired deadline **only in
that newly created replication protocol**, recording its parent protocol hash
and the new timestamp as above. Never edit a completed protocol or source to
resume results under different assumptions. Hierarchy summaries use
`hierarchy_surface --summarize`; temporal summaries use
`temporal_hierarchy_report`. Their embedded `--self-test` checks are independent
of the local response banks. Fine-grid and continuous replay protocols have
additional explicitly recorded mesh/particle budgets; they are not implied by
the finite replication recipe above.

The campaign environment used Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1,
PyTorch 2.14.1+cu130 and Matplotlib 3.11.2. The minimal requirements above are a
portable starting point, not a claim of bitwise equivalence. NumPy 2 or later is
required by `numpy.trapezoid`; the repository's older general minimum alone is
insufficient for these new metric scripts. GPU execution is optional for bank
construction; inference and all four core test modules can run on CPU.
