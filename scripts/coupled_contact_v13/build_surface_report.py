"""Build the completed extension report from audited, fixed-cohort results."""
from pathlib import Path
import hashlib
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/coupled-surface-v13'
STYLES = getSampleStyleSheet()
STYLES['BodyText'].fontSize = 9.5
STYLES['BodyText'].leading = 13
STYLES['BodyText'].spaceAfter = 8
STYLES['Heading1'].fontSize = 21
STYLES['Heading1'].leading = 25
STYLES['Heading1'].textColor = colors.HexColor('#173c50')
STYLES['Heading2'].fontSize = 12
STYLES['Heading2'].leading = 15
STYLES.add(ParagraphStyle(name='Small', fontSize=8, leading=10.5, spaceAfter=6))


def p(text, style='BodyText'):
    return Paragraph(escape(str(text)), STYLES[style])


def fmt(value):
    return 'n/a' if value is None else f'{value:.3f}'


def table(rows, widths):
    item = Table([[p(value, 'Small') for value in row] for row in rows], colWidths=widths, repeatRows=1)
    item.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4edf0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f7f8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    return item


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawString(42, 25, 'Coupled surface-contact v13 | Controlled simulation study')
    canvas.drawRightString(553, 25, str(document.page))
    canvas.restoreState()


def validate_evidence(protocol):
    """Refuse stale/failed audits or absent/mislabeled fixed clips."""
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts={}
    for name,required in [('independent-audit.json',['protocol.json','method-freeze.json','physics-development-qualification.json','summary.json','development-results.json','reserved-results.json','ablation-results.json']),('ablation-audit.json',['protocol.json','method-freeze.json','ablation-results.json','supplementary-protocol.json'])]:
        path=REPORT/name;audit=json.loads(path.read_text())
        if audit.get('passed') is not True or audit.get('errors') or audit.get('pending'):raise RuntimeError('Successful completed audit required: '+name)
        if name=='independent-audit.json' and audit.get('prior_preserved_files')!=405:raise RuntimeError('405 prior artifacts must pass preservation audit')
        for source in required:
            if audit.get('audited_report_sha256',{}).get(source)!=digest(REPORT/source):raise RuntimeError('Audit stale or missing source digest: '+name+'/'+source)
        artifacts[name]=digest(path)
    prior=json.loads((REPORT/'prior-preservation.json').read_text())['sha256']
    if len(prior)!=405 or any(not (ROOT/rel).exists() or digest(ROOT/rel)!=sha for rel,sha in prior.items()):raise RuntimeError('Prior405 artifacts changed since audit')
    artifacts['prior-preservation.json']=digest(REPORT/'prior-preservation.json')
    path=REPORT/'video-provenance.json';videos=json.loads(path.read_text());expected={f'{o}-{scene}.mp4':(o,scene) for o,scene in protocol['video_cases']}
    if len(videos)!=len(expected) or {Path(r['path']).name for r in videos}!=set(expected):raise RuntimeError('Fixed video coverage incomplete or duplicated')
    for video in videos:
        rel=Path(video['path'])
        if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Video artifact outside repository')
        if digest(ROOT/rel)!=video['sha256']:raise RuntimeError('Video artifact hash mismatch')
        obj,scene=expected[rel.name];base=ROOT/'runs/coupled-surface-v13'
        for file,key in [(base/'reference'/obj/scene/'trajectory.npz','reference_sha256'),(base/'evaluation'/obj/scene/'predictions.npz','prediction_sha256'),(base/'reference'/obj/scene/'complete.json','primary_status_sha256')]:
            if digest(file)!=video[key]:raise RuntimeError('Video input changed: '+key)
        if video['frames']!=61 or video['simulation_fps']!=100 or video['playback_fps']!=25:raise RuntimeError('Video timing differs from stated playback')
        stress=scene in protocol['stress_forecast_scenarios']
        if video.get('forecast_cohort')!=('stress' if stress else 'primary') or video.get('development_scope_qualified') is not (not stress):raise RuntimeError('Video cohort qualification mislabeled')
        if stress and 'unqualified' not in video.get('reference_display_label','').lower():raise RuntimeError('Stress clip lacks unqualified label')
        artifacts[str(rel)]=video['sha256']
    artifacts['video-provenance.json']=digest(path)
    return artifacts


def main():
    files = ['protocol.json', 'summary.json', 'ablation-results.json', 'material-recovery.json']
    data = {name: json.loads((REPORT/name).read_text()) for name in files}
    protocol, summary = data['protocol.json'], data['summary.json']
    audit_video_digests=validate_evidence(protocol)
    primary = summary['forecast_cohorts']['reserved']['primary']
    full = summary['forecast_cohorts']['reserved']['full']
    stress = summary['forecast_cohorts']['reserved']['stress']
    if primary['present'] != primary['expected'] or full['present'] != full['expected']:
        raise RuntimeError('Report requires every declared reserved forecast record, including failures')
    if any(row['present'] != row['expected'] for row in summary['reference_accounting']['levels'].values()):
        raise RuntimeError('Report requires every declared reference completion record, including failures')
    labels = {'fem': 'Inferred nonlinear FEM', 'rigid': 'Rigid + shared contact',
              'native': 'Adapted PhysTwin spring core', 'position_ridge': 'Position ridge',
              'oracle': 'True-parameter coarse FEM'}
    rows = [['Primary method', 'Complete', 'World RMSE (mm)', 'Shape RMSE (mm)']]
    for name, label in labels.items():
        record = primary['methods'][name]
        rows.append([label, f"{record['complete']}/{record['expected']}",
                     fmt(record['coordinate_mean_mm']), fmt(record['shape_mean_mm'])])
    story = [p('Position-based prediction with coupled deformable contact', 'Heading1'),
        p('Numerical qualification, parameter fitting and reserved cross-interaction evaluation'),
        p('We fit stiffness and friction from surface positions during a squeeze, shear and release, '
          'then predict new interactions using nonlinear deformation, rigid-body motion and explicit contact. '
          'Contact forces are computed from the scene; observed force labels are not inputs.'),
        p('Fixed primary comparison', 'Heading2'),
        p(f"The primary group contains {primary['expected']} reserved forecasts across eight objects: "
          + ', '.join(primary['scenarios']) + '. Membership was fixed before fresh outcomes. '
          'All declared cases remain, including numerical-reference failures and failed predictions.'),
        table(rows, [215, 60, 110, 110]), Spacer(1, 8),
        p('World error measures positions in the shared scene. Shape error removes a best-fitting proper '
          'rigid transformation per frame. Means for incomplete methods are conditional on completion; '
          'the coverage column is part of the result. Repeated interactions share eight objects.'),
        p(f"Joint calibration/forecast numerical checks pass for {primary['reference_qualified']}/"
          f"{primary['expected']} primary cases. This coverage does not change the primary denominator.")]
    pairs = primary['fem_comparisons'].get('rigid')
    if pairs:
        story.append(p(f"Against rigid contact, inferred FEM wins {pairs['wins']}, loses {pairs['losses']} "
            f"and ties {pairs['ties']} across {pairs['pairs']} completed pairs; mean paired error reduction "
            f"is {fmt(pairs['mean_gain_mm'])} mm."))
    supplements = summary['matched_ablation']['forecast_cohorts']['reserved']['primary']['methods']
    sup_rows = [['Matched primary control', 'Complete', 'Paired', 'World RMSE (mm)', 'FEM gain (mm)']]
    for name, label in [('nominal', 'Fixed nominal FEM'), ('vertex', 'Vertex-contact formulation')]:
        record = supplements[name]
        sup_rows.append([label, f"{record['complete']}/{record['expected']}", str(record['paired_cases']),
                         fmt(record['mean_coordinate_mm']), fmt(record['surface_fem_gain_mm'])])
    story.extend([table(sup_rows, [175, 60, 50, 105, 105]),
        p('Vertex contact is refit on the same observations and matches scene, mass and mesh. Its damping '
          'mass differs from the surface formulation, so this does not isolate quadrature alone.', 'Small'), PageBreak()])

    story.extend([p('Method and information available', 'Heading1'),
        p('1. Observe and fit', 'Heading2'),
        p(f"The context contains {protocol['observation_nodes']} surface nodes with "
          f"{protocol['observation_noise_std_m']*1000:g} mm Gaussian position noise and known tool poses. "
          f"A fixed grid of {len(protocol['fem_young_grid'])} stiffness values and "
          f"{len(protocol['friction_grid'])} friction values is scored by calibration position error. "
          'Selection never uses future target positions. Initial geometry, density, gravity, Poisson ratio, '
          'viscosity and collider geometry are known.'),
        p('2. Predict the coupled scene', 'Heading2'),
        p('The deformable object uses compressible Neo-Hookean tetrahedral FEM, objective viscous damping '
          'and consistent P1 mass. Rigid bodies have separate position and momentum states and exact '
          'uniform-volume inertia. Contact applies equal and opposite reactions. Future commanded tool '
          'positions drive the rollout; the future position of a free rigid body is predicted.'),
        p(summary['contact_representation']),
        p('3. Compare matched explanations', 'Heading2'),
        p('Rigid contact fits friction while suppressing deformation. The oracle diagnostic receives true '
          'stiffness and friction but retains the coarse predictor. Fixed nominal FEM uses 6000 Pa and '
          'friction 0.35. Position regressions are weaker controls without scene collision geometry. '
          'The native baseline uses the authors\' unchanged PhysTwin spring-force core within this study\'s '
          'graph, masses, contact, integration and global parameter search. It is not the full authors\' method.'),
        p('Parameter recovery is a separate outcome', 'Heading2')])
    recovery = data['material-recovery.json']['splits']['reserved']
    story.append(p(f"FEM parameter fits complete for {recovery['complete']}/{recovery['expected']} reserved objects. "
        f"Mean absolute relative stiffness error: {fmt(None if recovery['mean_absolute_relative_young_error'] is None else 100*recovery['mean_absolute_relative_young_error'])}%. "
        f"Mean absolute friction error: {fmt(recovery['mean_absolute_friction_error'])}. "
        'These are finite-grid point estimates, not a posterior or calibrated uncertainty. '
        'Fine/coarse model discrepancy can bias effective parameter estimates even when motion improves.'))
    story.extend([p('Transfer scope', 'Heading2'),
        p('The reserved materials are new combinations inside the declared parameter range. Forecast '
          'interactions differ from the calibration interaction, but their families were used during '
          'development. This is not unseen-geometry, unseen-material-range or real-world generalization.'), PageBreak()])

    grid = [['Scenario', 'Reference cells / axis', 'Companion cells / axis', 'Base step (us)']]
    for scene, settings in protocol['reference_settings_by_scenario'].items():
        grid.append([scene + (' (stress)' if scene in protocol['stress_forecast_scenarios'] else ''), settings['cells'][0], settings['companion_cells'][0], settings['dt']*1e6])
    account = summary['reference_accounting']
    story.extend([p('Numerical evidence and retained stress cases', 'Heading1'),
        p(f"The common predictor has {protocol['coarse_cells'][0]} cells per axis and a "
          f"{protocol['dt_coarse']*1e6:g} microsecond step. Reference choices were frozen from development evidence."),
        table(grid, [175, 105, 110, 105]), Spacer(1, 8),
        p(f"All {account['expected_total']} declared reference jobs are accounted for: a primary, a half-step "
          'run and a distinct-mesh companion for every object and scenario. Spatial comparisons use common '
          '27-point and 2678-surface-point probes. Fixed gates are 1 mm world-coordinate RMS, 0.5 mm aligned '
          'shape RMS, 1 mm body-center RMS, 0.25 mm temporal RMS, at most 1 mm penetration and positive '
          'element volume with J > 0.02. Primary qualification requires time-step checks at both development resolutions.'),
        p('These are two-resolution consistency checks, not a proof of continuum convergence or measured '
          'physical truth. Per-case flags are retained, and missing or failed comparisons are unqualified.'),
        p('Drop diagnosis', 'Heading2')])
    if stress['expected']:
        story.append(p('The drop family remains development-unqualified and is excluded from the fixed primary '
            f"scope by a prospective decision. All {stress['expected']} reserved drop forecasts remain in "
            f"the full {full['expected']}-case secondary accounting. An individual fresh numerical pass "
            'does not promote a stress case into the primary group. No reliable-drop claim is made.'))
    else:
        story.append(p('The adopted drop geometry and discretization passed the declared development gates. '
            'Earlier sharp-edge and rounded-support failures remain archived; success of an engineered '
            'replacement does not retrospectively qualify the original sharp support.'))
    story.extend([p('The original sharp support showed substantial mesh sensitivity. A rounded capsule rail '
        'was tested as a separately declared engineering redesign with the same width and top height. '
        'Its underside differs from the box; it is not merely isolated corner smoothing. Contact integration '
        'and time resolution were checked separately from spatial resolution. All failed trials are retained.'),
        p('Software and preservation', 'Heading2'),
        p('Independent verification reconstructs observations, selection losses, metrics, fixed cohort '
          'membership and reference quality. Frozen sources, protocol, compiler artifacts and evidence '
          'digests make the selected experiment reviewable. The 405 recorded prior artifacts remain unchanged.'), PageBreak()])

    story.extend([p('What this establishes and what remains', 'Heading1'),
        p('The concrete contribution of this campaign is a more carefully qualified coupled simulation '
          'comparison: material-surface contact, consistent inertia, position-only fitting, explicit '
          'free-body response and honest accounting of failed references. Prediction gains are measured '
          'against matched controls on the fixed primary group.'),
        p('Public benchmark transfer', 'Heading2'),
        p(summary['public_benchmark_scope']),
        p('The 21-case development pilot changed only volume-based rigid pose/inertia in an existing '
          'release predictor. It did not transfer the full coupled deformation model. Exact old replay '
          'and unchanged observed prefixes were independently checked. A broader public-dataset adapter '
          'still needs compatible geometry, units, interaction inputs and its own numerical validation.'),
        p('Remaining scientific limits', 'Heading2'),
        p('Known scene geometry and gravity are substantial inputs. Unknown support geometry, arbitrary '
          'boundary fields, multiple deformable bodies, image reconstruction, calibrated uncertainty and '
          'uncertainty-driven probing are not established by this experiment. The adapted native spring '
          'comparison cannot support an official PhysTwin or conference benchmark ranking.'),
        p('Videos and reproducibility', 'Heading2'),
        p('Three video cases were fixed before outcome inspection. They share world coordinates and camera '
          'motion, display method failures and numerical-reference status, and use no deformation '
          'exaggeration. Playback is four times slower than physical time. Stress clips are visibly labeled.'),
        p('Code: scripts/coupled_contact_v13. Frozen protocol, cohort tables, numerical checks and video '
          'provenance: reports/coupled-surface-v13. Historical study: reports/coupled-contact-v13. '
          'Raw trajectories, datasets and checkpoints remain local.'),
        p('Read the fixed primary table together with completion counts, matched paired comparisons and '
          'numerical qualification coverage. The full and stress tables remain available; no failed '
          'case was silently removed to improve the headline result.')])
    destination = ROOT/'papers/coupled-surface-v13/report.pdf'
    destination.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=42, bottomMargin=49, title='Position-based prediction with coupled deformable contact').build(
        story, onFirstPage=footer, onLaterPages=footer)
    provenance = dict(path=str(destination.relative_to(ROOT)), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((REPORT/name).read_bytes()).hexdigest() for name in files},
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_video_artifact_sha256=audit_video_digests,
        scope='Completed extension; must render and visually review before delivery')
    (REPORT/'paper-build.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
