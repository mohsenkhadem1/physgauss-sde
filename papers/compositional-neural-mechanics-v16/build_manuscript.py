"""Build an evidence-linked working manuscript; missing studies stay pending.

Run with runs/phystwin-env/bin/python. Reports are read, never modified.
"""
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports/compositional-neural-mechanics-v16'
OUT = ROOT / 'output/pdf/compositional-neural-mechanics-v16.pdf'
EVIDENCE = {}
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('TitleCustom', parent=styles['Title'], fontName='Helvetica-Bold',
                         fontSize=22, leading=26, textColor=colors.HexColor('#15394c'), spaceAfter=16))
styles.add(ParagraphStyle('TextCustom', parent=styles['BodyText'], fontSize=10, leading=14,
                         spaceAfter=9))
styles.add(ParagraphStyle('CaptionCustom', parent=styles['BodyText'], fontSize=8.5, leading=11,
                         textColor=colors.HexColor('#425466'), spaceAfter=10))
styles.add(ParagraphStyle('EquationCustom', parent=styles['BodyText'], fontName='Courier',
                         fontSize=9, leading=14, backColor=colors.HexColor('#eff4f7'),
                         borderPadding=8, spaceBefore=8, spaceAfter=16))
styles.add(ParagraphStyle('TableCustom', parent=styles['BodyText'], fontSize=8, leading=10))
story = []


def read(name):
    p = REPORTS / name
    EVIDENCE[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def para(text, style='TextCustom'):
    story.append(Paragraph(text, styles[style]))


def heading(text):
    story.append(Paragraph(text, styles['Heading2']))


def table(rows, widths):
    cells = [[Paragraph(escape(str(c)), styles['TableCustom']) for c in r] for r in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4edf2')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#668394')),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#b2c2cc')),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
    ]))
    story.extend([t, Spacer(1, 10)])


def page():
    story.append(PageBreak())


para('Compositional Neural Mechanics<br/>with Observable Physical Uncertainty', 'TitleCustom')
para('Working research manuscript | Simulation study | 2 November 2026', 'CaptionCustom')
heading('Abstract')
para('A reconstruction can reproduce observed motion while predicting the wrong response to a new interaction. '
     'We study a representation that separates reusable bulk material energies from relational attachment energies, '
     'and composes both through a nonlinear equilibrium solver. Scene-specific continuous descriptors encode material '
     'heterogeneity, support stiffness, nonlinearity, location and extent. Prescribed tool positions and orientations '
     'are inputs; deformation and reaction forces are outputs. We distinguish forward-law transfer with known '
     'descriptors from inference with unknown descriptors, and distinguish numerical convergence from uncertainty calibration.')
para('The implementation preserves an exact common-stiffness scaling symmetry: with zero body load, scaling all '
     'stiffnesses changes forces but leaves every position-controlled equilibrium unchanged. Known gravity and density '
     'can supply the missing load reference. Development experiments demonstrate accurate nonlinear response, '
     'improved inverse optimization, and useful refinement from an additional observation. A reserved composition '
     'and geometry evaluation and an independent numerical posterior check are reported below when their audits finish. '
     'The present scope is quasistatic tracked-point mechanics with known geometry and categorical law families; '
     'it does not establish a complete video-to-simulator system or superiority to published vision systems.')
heading('1. Scientific question')
para('The target is counterfactual physical accuracy: given an observation history, can a model predict the deformation '
     'and reactions under a previously unseen command? Rendering agreement measures a different quantity. '
     'Multiple physical explanations can produce the same visible shape, and a better image fit need not reduce '
     'error in future forces. We therefore report physical outputs separately and evaluate observations through '
     'their effect on future predictions.')
heading('Contribution under test')
para('We test whether shared local laws, composed by the same mechanical operator, transfer across material-support '
     'combinations and geometry more systematically than direct displacement prediction. A second question is whether '
     'scene inference and action-conditioned refinement preserve physical ambiguity while improving useful predictions. '
     'Neural constitutive laws, differentiable mechanics, Gaussian support fields and active sensing are established '
     'ingredients; the proposed contribution is their composition and intervention-based validation.')

page()
heading('2. Representation and mechanics')
para('Material coordinates X remain attached to the object. Its state is x(X)=X+u(X), with deformation gradient '
     'F=I+grad u. Geometry is a mapped cuboid with optional taper and arch. The present material field is isotropic '
     'and hyperelastic; spatial variation is expressed in material coordinates rather than a fixed world-space field.')
para('W(F, [s,z]) = exp(s) W_shared(F,z)<br/>P = dW/dF', 'EquationCustom')
para('The bulk network uses rotational invariants and positive coefficients. Factoring out the common stiffness '
     'exp(s) enforces the scaling identity exactly. This construction provides objectivity and a consistent energy-stress '
     'relation; we do not claim a general polyconvexity guarantee. The training families are Mooney-Rivlin and Yeoh-like '
     'responses, with continuous scale and shape descriptors.')
para('A separate shared attachment energy depends on relative displacement between a material point and its commanded '
     'tool target. Gaussian footprints specify location and width. The reference-area-normalized energy includes '
     'positive stiffness and nonlinear shape parameters, and supports polynomial and saturating response families. '
     'These are bilateral compliant attachments, not nonpenetrating contact constraints.')
para('Pi(u) = integral W(F,z_m) dV<br/>      + sum integral psi(delta(u,a),z_b) dA<br/>      - integral rho g . u dV<br/>R(u,z,a) = dPi/du = 0', 'EquationCustom')
para('A quadratic finite-element discretization, Newton iterations and line search solve equilibrium. The implementation '
     'checks residuals and positive deformation Jacobians. Gravity and density remain explicit body-load inputs. '
     'Forces and torques on the prescribed tools are derived from the same energy and solved state.')
heading('Scene inference')
para('The 15 continuous unknowns are bulk log stiffness, shape and heterogeneity amplitude; two attachment log stiffnesses '
     'and nonlinearities; four footprint center coordinates; and four log widths. Geometry, categorical bulk/interface '
     'family and heterogeneity pattern are conditioned on. The learned law weights stay fixed while these scene '
     'descriptors are fitted to noisy tracked surface positions. Reaction observations enter only in explicitly labeled '
     'measurement variants.')
para('H du/dz = -dR/dz<br/>L(z) = 0.5 ||r_observation(z)||^2 + 0.5 ||r_prior(z)||^2', 'EquationCustom')
para('Implicit sensitivities differentiate the converged mechanical state. A bounded Gauss-Newton trust-region solver '
     'uses the full residual Jacobian and its scaling to improve optimization conditioning. Its stopping status is '
     'reported separately from parameter accuracy, predictive accuracy and posterior validity.')

page()
heading('3. Observability and refinement')
para('<b>Common-scale ambiguity.</b> If gravity and other prescribed body forces vanish, multiplying every bulk and '
     'attachment stiffness by c multiplies the entire energy by c. The equilibrium displacement is unchanged, '
     'whereas the reactions multiply by c. All rendered views of that state are therefore unchanged. This ambiguity '
     'persists across every prescribed-position experiment in this model; additional cameras cannot remove it.')
para('Known gravity with known density introduces a fixed reference load and changes this conclusion: stiffness affects '
     'visible sag. If density is also free to scale with stiffness, a joint ambiguity remains. These assumptions matter '
     'when claiming that position-only observations identify physical properties.')
figure = REPORTS / 'stiffness-observability-development.png'
if figure.exists():
    EVIDENCE[str(figure.relative_to(ROOT))] = hashlib.sha256(figure.read_bytes()).hexdigest()
    from PIL import Image as PILImage
    with PILImage.open(figure) as im:
        w, h = im.size
    story.append(Image(str(figure), width=468, height=468*h/w))
    para('Development likelihood slices at fixed relative stiffnesses. These are not marginalized posteriors or '
         'confidence intervals. The zero-gravity and known-gravity experiments use different noise seeds.', 'CaptionCustom')
heading('Functional uncertainty')
para('A local Gaussian approximation uses the observation Jacobian and an explicit prior precision. Candidate actions '
     'are ranked by their expected reduction in variance of a specified future response, rather than total parameter '
     'variance. The target commands are public inputs; their response labels are unavailable during selection.')
para('C = (J_obs^T J_obs + P_prior)^(-1)<br/>C_after(a) = (C^(-1) + J_a^T J_a)^(-1)<br/>score(a) = trace[A (C - C_after(a)) A^T]', 'EquationCustom')
para('This approximation does not automatically account for nonlinear multimodality, active bounds or model discrepancy. '
     'The development bands set model discrepancy to zero and are labeled nominal. A separate small reference posterior '
     'tests integration convergence before any claim that its intervals are trustworthy. Empirical coverage and '
     'fixed/random acquisition controls remain required.')

page()
heading('4. Development evidence')
v = read('validation-runtime-recovery-results.json')
rows = [['Forward model', 'Position RMSE (mm)', 'Force RMSE (N)', 'Torque RMSE (Nm)']]
for key, label in [('shared_laws', 'Shared laws'), ('neo_hookean', 'Neo-Hookean')]:
    r = v['summary'][key]
    rows.append([label, f"{r['displacement_coordinate_rmse_m']*1e3:.4f}",
                 f"{r['force_coordinate_rmse_N']:.5f}", f"{r['torque_coordinate_rmse_Nm']:.6f}"])
table(rows, [142, 108, 108, 110])
para('Four validation objects, six commands each; both models use known descriptors and n=4. '
     'One original runtime failure was replayed in an isolated process with identical settings; the failure record '
     'is retained. These validation numbers are not the reserved test.', 'CaptionCustom')
g = read('gauss-newton-development-results.json')
if read('gauss-newton-final-independent-audit.json').get('status') != 'passed':
    raise ValueError('Gauss-Newton table requires independent audit')
rows = [['Object / model', 'Original position (mm)', 'Gauss-Newton (mm)', 'GN force (N)']]
for r in g['rows']:
    rows.append([f"{r['world']} / {r['model']}",
                 f"{1e3*r['original']['mean_heldout_errors']['displacement']:.4f}",
                 f"{1e3*r['gauss_newton']['mean_heldout_errors']['displacement']:.4f}",
                 f"{r['gauss_newton']['mean_heldout_errors']['forces']:.5f}"])
table(rows, [142, 108, 108, 110])
para('Full 15-parameter development inversion, two observed objects and two future commands each. '
     'All four Gauss-Newton fits stopped successfully by ftol, versus zero original fits within their budgets. '
     'All motion errors decreased; force errors increased for object 0 and decreased for object 1. '
     'This is an optimization comparison, not an independent generalization result.', 'CaptionCustom')
heading('Observed failure modes and repairs')
para('A narrow-slip attachment training distribution produced a 4.065 mm same-mesh error in an exposed stretching case. '
     'A separately documented warm-start, wide-slip training run reduced that case to 0.079 mm. A failed alternative '
     'wide-range run is retained. The repaired law was frozen before reserved testing; the exposed case is not counted '
     'as evidence of unseen-scene transfer.')
para('Development mesh checks also showed that n=4 was inadequate for several extreme reference cases; targeted n=6 '
     'to n=8 checks passed with field differences of 2.98-8.15 micrometres. The reserved benchmark consequently uses '
     'a fixed n=8 analytic reference. That choice is supported by development tests, not individual convergence '
     'certification for every reserved scene.')

page()
heading('5. Reserved forward transfer')
para('Training uses 54 objects from 18 geometry-material-interface-heterogeneity cells and six commands, plus four '
     'validation objects. Six factor combinations are withheld while each pair of factors remains represented. '
     'The reserved test has six withheld-composition objects and six draws on two new geometries, each evaluated '
     'under four new command directions or combinations. Test identities, checkpoints, source hashes and methods '
     'are committed before generating results.')
para('All forward methods receive oracle descriptors. Comparisons include the matched analytic family at n=4, '
     'small-strain-matched Neo-Hookean, learned laws at n=4, and custom direct point predictors trained for 6,000 '
     'and 30,000 steps. The direct predictors receive numeric law fingerprints from the same learned laws; this '
     'provides a shared physical prior but is not computationally identical to full solver access. These custom '
     'networks are not reproductions of PhysTwin or MeshGraphNet.')
primary = REPORTS / 'oracle-reserved-forward-results.json'
audit = REPORTS / 'oracle-final-independent-audit.json'
if primary.exists() and audit.exists() and read(audit.name).get('passed'):
    result = read(primary.name)
    rows = [['Cohort / model', 'Position (mm)', 'Nonrigid (mm)', 'Force (N)']]
    for r in result['summaries']:
        if not r['complete']:
            rows.append([r['cohort']+' / '+r['method'], 'INCOMPLETE', '-', '-'])
            continue
        x = r['object_mean_scores']
        rows.append([r['cohort'].replace('reserved_', '')+' / '+r['method'],
                     f"{1e3*x['displacement_coordinate_rmse_m']:.4g}",
                     f"{1e3*x['canonical_nonrigid_coordinate_rmse_m']:.4g}",
                     f"{x['force_coordinate_rmse_N']:.4g}"])
    table(rows, [222, 82, 82, 82])
    para(f"All {len(result['rows'])} declared rows audited; {result['failed_predictions']} failed outputs. "
         'Means are averaged by object. Confidence intervals resample objects, not commands. '
         'Geometry conclusions are conditional on two new shape instances.', 'CaptionCustom')
else:
    para('<b>Reserved results pending independent completion audit.</b> No partial-cohort averages are reported.')
heading('Preprocessing control')
para('A source-only review identified that training-constant command coordinates were divided by a standard '
     'deviation floor of 1e-8, causing extreme features for new directions. The primary frozen baselines are '
     'retained. A supplementary control replaces only exactly zero-mean, zero-variance input scales with fixed '
     'physical units; weights and all training predictions remain unchanged. Its predictions are committed '
     'before reading reference outputs. Final interpretation must include this control rather than crediting '
     'the mechanics architecture for a normalization defect.')

page()
heading('6. Observation refinement and uncertainty checks')
action_file = REPORTS / 'local-action-development-results.json'
if action_file.exists():
    if not read('local-action-final-independent-audit.json').get('passed'):
        raise ValueError('Action table requires independent audit')
    a = read(action_file.name)
    rows = [['Policy', 'Mean position (mm)', 'Mean nominal 90% coverage']]
    for policy in ['initial', 'fixed_stretch', 'fixed_right_bend', 'fixed_opposed_twist', 'random', 'target_task_variance']:
        rr = [r for r in a['rows'] if r['policy'] == policy]
        if len(rr) != 4 or any(r['state'] != 'complete' for r in rr):
            rows.append([policy, 'INCOMPLETE', '-'])
        else:
            rows.append([policy, f"{1e3*sum(r['position_coordinate_rmse_m'] for r in rr)/4:.4f}",
                         f"{100*sum(r['nominal90_coordinate_coverage'] for r in rr)/4:.1f}%"])
    table(rows, [210, 120, 138])
    para('Two exposed development objects and two future commands per object. Each non-initial policy receives '
         'exactly one additional observation. The variance policy selected stretch for both objects and therefore '
         'ties fixed stretch. Pointwise coordinates are correlated; these coverage fractions are descriptive, '
         'not independent calibration trials or evidence of a superior policy.', 'CaptionCustom')
heading('Reference posterior')
p = REPORTS / 'reference-posterior-results.json'
if (REPORTS/'reference-posterior-recovery-results.json').exists():
    p = REPORTS/'reference-posterior-recovery-results.json'
if p.exists():
    r = read(p.name)
    para('The three-parameter conditional reference posterior is '+
         ('numerically qualified under its declared checks.' if r.get('qualified') else
          'not yet numerically qualified. No trustworthy interval claim is made from the current run.'))
para('This reference calculation exploits the exact zero-gravity common-scale identity: it solves in two relative '
     'stiffness coordinates and integrates the common scale separately with the original bounded prior. '
     'Grid refinement, conditional integration and proposal-width checks are required. It is a diagnostic of '
     'a small conditional inverse problem, not a 15-dimensional posterior validation.')
para('A separately frozen six-object inverse study is running with unknown continuous descriptors, both categorical '
     'bulk and attachment families represented, n=4 analytic observations and n=2 inference. It compares fitted analytic '
     'and learned laws, with fixed, random and task-variance policies at equal observation budgets. '
     'The discretization mismatch intentionally tests whether the same-mesh development result survives forward-model error.')
heading('Contact extension')
para('A separate normal-penalty operator evaluates signed distance to prescribed planes and spheres and returns '
     'consistent rigid reactions. Plane contact passed the development mesh and quadrature checks. Sphere and '
     'combined cases still showed mesh-sensitive fields or reactions, and a tiny contact patch was missed by '
     'coarse quadrature. These failures remain explicit. Finite penetration, absent friction and prescribed rigid '
     'geometry prevent treating this extension as a general coupled-contact result.')

page()
heading('7. Relation to vision-based physical reconstruction')
para('<b>PhysTwin [1]</b> reconstructs deformable digital twins from sparse multiview videos using shape priors, '
     'spring-mass mechanics and Gaussian appearance. Our current tracked-point, known-geometry study isolates '
     'local-law composition and inference; it is not an end-to-end comparison with that reconstruction system.')
para('<b>PhysGS [2]</b> maintains Bayesian physical-property beliefs over Gaussian representations and refines them '
     'from observations. Our evaluation instead targets future mechanical responses and explicit observability '
     'under a coupled equilibrium model. Bayesian property estimation itself is not a novel ingredient here.')
para('<b>PhysHO [3]</b> incorporates neural constitutive modeling into human-object interaction simulation, while '
     '<b>NeuSpring [4]</b> uses neural spring-property fields. These works preclude claiming that a neural material '
     'field or constitutive component alone is new. Our hypothesis concerns reusable bulk/interface composition '
     'and its validation under unknown scene descriptors and new interventions.')
heading('8. Scope and next evidence')
para('The implementation supports useful pieces of the proposed architecture: reusable bulk and attachment laws, '
     'position-controlled equilibrium, scene inference, structural observability and task-aware action scoring. '
     'Current evidence is restricted to mapped cuboids, known categorical families, known geometry and quasistatic '
     'elasticity. It does not establish arbitrary unseen material families, topology changes, soft-soft contact, '
     'friction, impacts, learned perception, Gaussian rendering quality or new-view acquisition.')
para('The next decisive evidence is a separately untouched inverse cohort, with all continuous material/support '
     'descriptors hidden, matched observation budgets and strong fitted analytic and fixed-action controls. '
     'Reference discretization error and learned-model discrepancy must be separated from latent uncertainty. '
     'Existing repository benchmark results remain preserved but cannot be attributed to this new representation '
     'without adapting and rerunning it.')
heading('References')
refs = [
    ('[1] H. Jiang et al. PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos. ICCV 2025.',
     'https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_PhysTwin_Physics-Informed_Reconstruction_and_Simulation_of_Deformable_Objects_from_Videos_ICCV_2025_paper.pdf'),
    ('[2] S. Chopra et al. PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation. CVPR 2026.',
     'https://openaccess.thecvf.com/content/CVPR2026/html/Chopra_PhysGS_Bayesian-Inferred_Gaussian_Splatting_for_Physical_Property_Estimation_CVPR_2026_paper.html'),
    ('[3] S. Jiang and G. H. Lee. PhysHO. CVPR 2026. Project and paper.', 'https://suezjiang.github.io/physho/'),
    ('[4] Q. Xu et al. NeuSpring. arXiv:2511.08310, 2025.', 'https://arxiv.org/abs/2511.08310'),
]
for label, url in refs:
    para(escape(label)+' <link href="'+url+'" color="#245b7a">Primary source</link>.', 'CaptionCustom')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#bacbd4'))
    canvas.line(54, 44, 558, 44)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526673'))
    canvas.drawString(54, 31, 'Compositional Neural Mechanics | Working manuscript | v16')
    canvas.drawRightString(558, 31, str(doc.page))
    canvas.restoreState()


if __name__ == '__main__':
    for name in ('wide-interface-equilibrium-qualification.json', 'nonlinear-range-fine-qualification.json',
                 'interface-wide-range-results.json', 'interface-wide-range-warm-results.json',
                 'contact-development-interpretation.json', 'coupled-scale-control.json',
                 'gravity-inference-pilot-results.json', 'scaling-identity-ablation-results.json'):
        if (REPORTS/name).exists():
            read(name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=(612, 792), rightMargin=72, leftMargin=72,
                            topMargin=48, bottomMargin=60,
                            title='Compositional Neural Mechanics with Observable Physical Uncertainty',
                            author='Research working draft')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    manifest = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_sha256': EVIDENCE, 'output': str(OUT.relative_to(ROOT)),
                'output_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
                'scope': 'Working manuscript; pending tests are not presented as completed evidence.'}
    (Path(__file__).parent/'manuscript-evidence.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT)
