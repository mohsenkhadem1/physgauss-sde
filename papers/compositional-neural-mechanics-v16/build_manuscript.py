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
para('The attachment-only model preserves an exact common-stiffness scaling symmetry: with zero body load, scaling all '
     'stiffnesses changes forces but leaves every position-controlled equilibrium unchanged. Known gravity and density '
     'can supply the missing load reference. Development experiments demonstrate accurate nonlinear response, '
     'improved inverse optimization, and useful refinement from an additional observation. On six withheld-composition '
     'objects and six draws on new geometries, shared-law position error is 0.0181 and 0.0183 mm, versus 3.09 and '
     '3.29 mm for the stabilized 30,000-step custom direct baseline with the same oracle descriptors. On six separate '
     'objects with 15 hidden continuous descriptors, one additional stretch observation reduces shared-law motion '
     'error from 0.339 to 0.080 mm and force error from 0.096 to 0.029 N without force measurements. The adaptive '
     'policy ties fixed stretch; nominal uncertainty remains sensitive to model discrepancy. Frozen laws also '
     'compose with normal contact in two known-descriptor development cases. A separate three-parameter '
     'continuous numerical reference passes declared surrogate-validation and integration checks. '
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
para('<b>Common-scale ambiguity.</b> In the attachment-only model with no additional fixed-stiffness energy terms, if gravity and other prescribed body forces vanish, multiplying every bulk and '
     'attachment stiffness by c multiplies the entire energy by c. The equilibrium displacement is unchanged, '
     'whereas the reactions multiply by c. All rendered views of that state are therefore unchanged. This ambiguity '
     'persists across every prescribed-position experiment in this model; additional cameras cannot remove it.')
para('With fixed-penalty contact, this identity requires co-scaling the penalty or imposing ideal hard-contact '
     'constraints. Numerical penalty stiffness is not physical scale evidence. The contact study uses known '
     'gravity and density and does not infer contact.')
para('Known gravity with known density introduces a fixed reference load and changes this conclusion: stiffness affects '
     'visible sag. If density is also free to scale with stiffness, a joint ambiguity remains. These assumptions matter '
     'when claiming that position-only observations identify physical properties.')
figure = REPORTS / 'stiffness-observability-development.png'
if figure.exists():
    EVIDENCE[str(figure.relative_to(ROOT))] = hashlib.sha256(figure.read_bytes()).hexdigest()
    from PIL import Image as PILImage
    with PILImage.open(figure) as im:
        w, h = im.size
    story.append(Image(str(figure), width=444, height=444*h/w))
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
heading('4.1. Visual agreement and physical response')
visual = read('visual-bridge-development-results.json')
visual_audit = read('visual-bridge-final-independent-audit.json')
if not visual_audit.get('passed') or visual_audit['result_sha256'] != EVIDENCE[str((REPORTS/'visual-bridge-development-results.json').relative_to(ROOT))]:
    raise ValueError('Visual table requires exact independently audited results')
para('A deterministic calibrated renderer uses the same solved mechanical field as the force and position '
     'predictions. Surface color is fixed in material coordinates, with no learned image correction. '
     'We evaluate three fixed views using foreground-union RGB error, joint-visible depth error and silhouette '
     'intersection-over-union (IoU); missing and extra pixels remain recorded separately.')
figure = REPORTS/'appearance-physical-ambiguity.png'
if figure.exists():
    EVIDENCE[str(figure.relative_to(ROOT))] = hashlib.sha256(figure.read_bytes()).hexdigest()
    read('appearance-physical-ambiguity.json')
    story.append(Image(str(figure), width=468, height=154.44))
    para('Exact zero-body-load counterexample: co-scaling bulk and support stiffness preserves equilibrium '
         'and all three full RGB/depth views while doubling reactions. Display crops are identical. '
         'This is a controlled mathematical ambiguity, not a claim about every gravity-loaded scene.', 'CaptionCustom')
rows = [['Object / prediction', 'Position mm', 'Force N', 'RGB RMSE', 'Depth mm', 'IoU']]
for item in visual['comparisons']:
    metrics = item['physical']
    views = list(item['views'].values())
    mean = lambda name:sum(v[name] for v in views)/len(views)
    rows.append([str(item['world'])+' / '+('initial' if 'initial' in item['method'] else 'after stretch'),
                 f"{metrics['position_coordinate_rmse_m']*1e3:.4f}",
                 f"{metrics['reaction_force_coordinate_rmse_N']:.5f}",
                 f"{mean('rgb_foreground_union_rmse'):.4f}",
                 f"{mean('depth_joint_rmse_m')*1e3:.4f}",
                 f"{mean('silhouette_iou'):.4f}"])
table(rows, [138, 66, 66, 66, 66, 66])
para('Two development objects, one declared held command per object. Analytic n=4 references versus learned '
     'n=2 predictions include discretization/model error, so these values differ from the earlier same-n=2 '
     'pilot. View means are descriptive; pixels are correlated. RGB is normalized to [0,1].', 'CaptionCustom')
para('Both objects improve in rendered and physical outputs after an extra stretch observation. Even the initial '
     'silhouettes have high IoU, illustrating why a favorable appearance metric cannot substitute for force '
     'or deformation validation. This bridge uses known geometry and synthetic appearance; it does not evaluate '
     'visual reconstruction, camera estimation or inference directly from images.')

page()
heading('5. Reserved forward transfer')
para('Training uses 54 objects from 18 combinations of geometry, material family, support layout and heterogeneity, '
     'with six commands per object, plus four '
     'validation objects. Six factor combinations are withheld while each pair of factors remains represented. '
     'The interface law family is sampled independently of these four factors. The reserved test has six '
     'withheld-composition objects and six draws on two new geometries, each evaluated '
     'under four new command directions or combinations. Test identities, checkpoints, source hashes and methods '
     'are committed before generating results.')
para('All forward methods receive oracle descriptors. Comparisons include the matched analytic family at n=4, '
     'small-strain-matched Neo-Hookean, learned laws at n=4, and custom direct point predictors trained for 6,000 '
     'and 30,000 steps. The direct predictors receive numeric law fingerprints from the same learned laws; this '
     'provides a shared physical prior but is not computationally identical to full solver access. These custom '
     'networks are not reproductions of PhysTwin or MeshGraphNet.')
heading('Runtime and preprocessing provenance')
para('The original forward run completed 240 of 288 reference/prediction rows; eight scene workers reached their '
     '1,800-second limits. A JSON tuple/list comparison then blocked aggregation. Calling the unchanged aggregator '
     'with committed JSON records preserved all 288 original rows, including 48 missing-output failures. '
     'A separately audited recovery recomputed only those 48 missing calls in a new directory, using identical '
     'frozen sources, checkpoints, n=4/n=8 discretizations, commands and the original runtime. All 288 merged rows '
     'are now complete; the original failure ledger remains available.')
para('A source-only normalization review found that command coordinates absent from training were divided by a '
     'standard-deviation floor of 1e-8. Separate stable controls replace only exactly zero training inputs with '
     'fixed physical-unit scales; weights, target normalization and all 324 training predictions remain unchanged. '
     'The two stable checkpoints were frozen before reference scoring. However, all 96 initial forecast writes '
     'failed because nested output directories were absent. A directory-only replay later produced 96 finite '
     'artifacts with the same frozen prediction code and checkpoints, after primary outcomes were visible. '
     'No retraining, preprocessing redesign or model selection followed those outcomes. Successful pre-truth '
     'forecast commitment is not claimed.')
para('The fair direct comparison uses these successful stable controls. The original unstable baselines are '
     'retained as preprocessing diagnostics, not evidence for an architectural advantage. Stable controls run '
     'on CPU with PyTorch 2.4; the original/recovered oracle models run on GPU with PyTorch 2.14. '
     'No cross-runtime bit-identity claim is made.', 'CaptionCustom')

page()
heading('5.1. Audited forward results and stable controls')
original_ledger = read('oracle-reserved-forward-results.json')
result = read('oracle-recovery-merged-results.json')
forward_audit = read('oracle-recovery-final-independent-audit.json')
stable = read('stable-zero-io-recovered-mixed-results.json')
stable_audit = read('stable-zero-io-final-independent-audit.json')
if not forward_audit.get('passed') or not stable_audit.get('passed') or not stable_audit.get('baseline_comparison_complete'):
    raise ValueError('Complete recovered forward and successful stable-control audits required')
for report_name, audit_value in [('oracle-recovery-merged-results.json',forward_audit),
                                 ('stable-zero-io-recovered-mixed-results.json',stable_audit)]:
    key=str((REPORTS/report_name).relative_to(ROOT))
    if audit_value['evidence_sha256'].get(key)!=EVIDENCE[key]:
        raise ValueError('Forward/control audit does not bind current result')
if len(original_ledger['rows'])!=288 or original_ledger['failed_predictions']!=48 or result['failed_predictions']!=0:
    raise ValueError('Preserved original/recovery census differs')
labels={'matched_analytic':'Matched analytic', 'neo_hookean':'Neo-Hookean', 'shared_laws':'Shared laws',
        'custom_direct_stable':'Direct 6k, stable', 'custom_direct_long_stable':'Direct 30k, stable'}
rows=[['Cohort / model','Position mm','Nonrigid mm','Force N']]
for cohort in ('reserved_composition','reserved_geometry'):
    for method in labels:
        summaries=stable['summaries'] if method.endswith('_stable') else result['summaries']
        item=next(r for r in summaries if r['cohort']==cohort and r['method']==method)
        if not item['complete'] or item['scored_commands']!=24:
            raise ValueError('No incomplete means in final forward table')
        x=item['object_mean_scores']
        rows.append([cohort.replace('reserved_','')+' / '+labels[method],
                     f"{1000*x['displacement_coordinate_rmse_m']:.4f}",
                     f"{1000*x['canonical_nonrigid_coordinate_rmse_m']:.4f}",
                     f"{x['force_coordinate_rmse_N']:.5f}"])
table(rows,[222,82,82,82])
para('Each cohort has six objects and four commands per object. All 24 commands per method are included. '
     'Position is coordinate RMSE; nonrigid error removes the best rigid alignment. Means weight objects equally. '
     'The geometry cohort contains six draws conditional on two new shape instances, not six independent geometry '
     'families. The matched analytic n=4 control also differs from the n=8 reference.', 'CaptionCustom')
para('Both stable direct budgets are reported without selecting the more favorable comparison. Shared laws retain '
     'a substantial advantage over the corrected custom direct predictors in this oracle-descriptor experiment. '
     'This is controlled evidence for composing reusable laws through mechanics; it is not a state-of-the-art '
     'vision benchmark or an official PhysTwin/MeshGraphNet comparison.')
heading('Preserved unstable-normalization diagnostic')
rows=[['Cohort','Original direct 6k mm','Original direct 30k mm']]
for cohort in ('reserved_composition','reserved_geometry'):
    entries=[next(r for r in result['summaries'] if r['cohort']==cohort and r['method']==m)
             for m in ('custom_direct','custom_direct_long')]
    rows.append([cohort.replace('reserved_',''),*[f"{1000*r['object_mean_scores']['displacement_coordinate_rmse_m']:.1f}" for r in entries]])
table(rows,[180,144,144])
para('These extreme values diagnose the frozen zero-variance preprocessing defect. They are intentionally excluded '
     'from the fair architectural comparison above. All original outputs, the failed-write ledger, and both '
     'directory-only and missing-call recovery records remain preserved.', 'CaptionCustom')

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
page()
heading('6.1. Unseen objects with unknown continuous parameters')
unknown_path = REPORTS/'unknown-action-reserved-results.json'
unknown_audit_path = REPORTS/'unknown-action-final-independent-audit.json'
if unknown_path.exists() and unknown_audit_path.exists():
    unknown = read(unknown_path.name)
    unknown_audit = read(unknown_audit_path.name)
    if not unknown_audit.get('passed') or unknown_audit['evidence_sha256'].get(str(unknown_path.relative_to(ROOT))) != EVIDENCE[str(unknown_path.relative_to(ROOT))]:
        raise ValueError('Unknown-object table requires exact independent audit')
    para('Six fresh continuous scene draws cover both bulk and attachment families on three familiar geometries. '
         'All 15 continuous descriptors are hidden; geometry, categorical families, gravity and density are known. '
         'Observations come from n=4 analytic simulations, while both fitted models use n=2. '
         'Every probe policy receives the initial position packet and exactly one additional packet; forces are not observed.')
    rows = [['Policy', 'Analytic mm', 'Shared mm', 'Analytic coverage', 'Shared coverage']]
    for policy in ('initial', 'fixed_stretch', 'fixed_right_bend', 'fixed_opposed_twist', 'uniform_random', 'target_task_variance'):
        selected = [next(s for s in unknown['summaries'] if s['model'] == model and s['policy'] == policy)
                    for model in ('analytic_fitted', 'shared_laws_fitted')]
        if any(s['complete_commands'] != 12 for s in selected):
            rows.append([policy, 'INCOMPLETE', 'INCOMPLETE', '-', '-'])
        else:
            rows.append([policy, *[f"{s['mean_coordinate_rmse_m']*1e3:.4f}" for s in selected],
                         *[f"{s['mean_coordinate_coverage']*100:.1f}%" for s in selected]])
    table(rows, [144, 81, 81, 81, 81])
    fits = unknown_audit['fit_counts']
    para(f"All 144 policy-command predictions are retained. {fits['optimizer_success']} of {fits['expected']} "
         f"fits met their stopping criteria; {fits['incomplete']} reached the evaluation limit. "
         'All 12 stretch fits selected across the two models converged. '
         'Incomplete fits are not silently excluded from metrics. Coverage is marginal coordinate coverage of '
         'nominal 90% local Gaussian bands; coordinates and commands are correlated.', 'CaptionCustom')
    para('The variance policy selected stretch for every object in both models, so it exactly ties that fixed '
         'control. Shared-law error after stretch is approximately 0.080 mm, close to fitted analytic mechanics. '
         'Both improve on all six objects relative to their initial estimates. This supports observation-based '
         'refinement, but not an adaptive advantage over the strongest fixed action.')
    para('Uncorrected bands become sharply overconfident after probing: the learned model covers only about one '
         'third of target coordinates. A separately frozen validation-only discrepancy estimate is evaluated as '
         'an additive prediction variance, with coverage, width and interval score reported together. '
         'It changes neither fitted parameters nor actions and is not a posterior-calibration guarantee.')
    band_path = REPORTS/'unknown-action-discrepancy-band-results.json'
    band_audit_path = REPORTS/'discrepancy-band-final-independent-audit.json'
    if band_path.exists() and band_audit_path.exists():
        bands, band_audit = read(band_path.name), read(band_audit_path.name)
        if not band_audit.get('passed') or band_audit['result_sha256'] != EVIDENCE[str(band_path.relative_to(ROOT))]:
            raise ValueError('Supplementary bands require exact independent audit')
        rows = [['Model', 'Coverage %', 'Half-width mm', 'Interval score mm']]
        for model, label in [('analytic_fitted', 'Analytic'), ('shared_laws_fitted', 'Shared')]:
            item = next(s for s in bands['summaries'] if s['model'] == model and s['policy'] == 'target_task_variance')
            rows.append([label,
                         f"{100*item['primary']['coverage']:.1f} / {100*item['supplementary']['coverage']:.1f}",
                         f"{1e3*item['primary']['mean_halfwidth_m']:.3f} / {1e3*item['supplementary']['mean_halfwidth_m']:.3f}",
                         f"{1e3*item['primary']['mean_interval_score_m']:.3f} / {1e3*item['supplementary']['mean_interval_score_m']:.3f}"])
        table(rows, [105, 110, 120, 133])
        para('Before / after adding the prospectively frozen validation MSE. Lower interval score is better; '
             'it penalizes both excessive width and missed targets. These six-object descriptive results '
             'do not establish population calibration.', 'CaptionCustom')
else:
    para('The separately frozen six-object inverse evaluation is pending its final independent audit.')

page()
heading('6.2. Reaction prediction from position observations')
reaction_path = REPORTS/'unknown-reaction-final-independent-audit.json'
if reaction_path.exists():
    reactions = read(reaction_path.name)
    read('unknown-reaction-secondary-results.json')
    reaction_key = str((REPORTS/'unknown-reaction-secondary-results.json').relative_to(ROOT))
    if not reactions.get('passed') or reactions['evidence_sha256'].get(reaction_key) != EVIDENCE[reaction_key]:
        raise ValueError('Secondary reaction table requires independent audit')
    para('A prospectively specified secondary replay evaluates reactions for the same fitted parameters and '
         'policy choices. No fitting or selection is repeated and no force observations are added. '
         'Reference and prediction torques use the same declared tool origins. All 108 replay states and '
         '144 policy-command reaction rows completed and passed independent checks.')
    rows = [['Policy', 'Analytic force N', 'Shared force N', 'Analytic torque Nm', 'Shared torque Nm']]
    for policy in ('initial', 'fixed_stretch', 'fixed_right_bend', 'fixed_opposed_twist', 'uniform_random', 'target_task_variance'):
        selected = [next(s for s in reactions['summaries'] if s['model'] == model and s['policy'] == policy)
                    for model in ('analytic_fitted', 'shared_laws_fitted')]
        if any(s['complete_commands'] != 12 for s in selected):
            rows.append([policy, 'INCOMPLETE', 'INCOMPLETE', '-', '-'])
        else:
            rows.append([policy, *[f"{s['force_coordinate_rmse_N']:.5f}" for s in selected],
                         *[f"{s['torque_coordinate_rmse_Nm']:.6f}" for s in selected]])
    table(rows, [144, 81, 81, 81, 81])
    para('Mean coordinate RMSE, equally averaged over two held commands per object and six objects. '
         'Primary fit-limit failures remain in the ledger; successful reaction replay does not make an '
         'incomplete parameter fit converged.', 'CaptionCustom')
    para('The learned model reduces reaction-force error from about 0.096 N initially to 0.029 N after stretch. '
         'Fixed twist has slightly lower force and torque errors than the selected stretch on this cohort. '
         'The acquisition criterion targets future positions; it is not a universally optimal criterion for '
         'every physical output. Fitted analytic and shared-law models have similar post-probe performance.')
else:
    para('The separate fixed-parameter reaction replay is pending its final independent audit.')

page()
heading('6.3. Numerical uncertainty and contact qualification')
heading('Reference posterior')
p = REPORTS / 'reference-posterior-extension-results.json'
if p.exists():
    r = read(p.name)
    a = read('reference-posterior-extension-final-independent-audit.json')
    if not a.get('passed') or a['report_sha256'] != EVIDENCE[str(p.relative_to(ROOT))]:
        raise ValueError('Reference posterior extension requires current independent audit')
    para(f'The completed extension evaluated {a["unique_nodes"]:,} unique mechanics nodes with '
         f'{a["failed_nodes"]} failed nodes. Its independent audit passed {a["checks"]:,} checks of '
         'saved responses, provenance and reported convergence decisions. The position-only V reference '
         'passes the declared numerical consistency tests; the position-plus-wrench VF reference does not.')
    c = a['recomputed_checks']['VF_proposal_std_doubling']
    para(f'For VF, doubling the proposal width changes an interval endpoint by '
         f'{c["interval_endpoint_change_in_sd"]:.6f} posterior standard deviations, above the fixed 0.05 '
         'limit, despite stable moments and evidence. The conditional scale CDF is accurately integrated, '
         'but the remaining two-dimensional ratio quadrature resolves sharp tail transitions poorly. '
         'No converged VF interval or posterior-calibration claim is made. Earlier failures remain preserved.')
else:
    para('The reference-posterior extension has not supplied completed audited evidence.')
para('This reference calculation exploits the exact zero-gravity common-scale identity: it solves in two relative '
     'stiffness coordinates and integrates the common scale separately with the original bounded prior. '
     'Grid refinement, conditional integration and proposal-width checks are required. It is a diagnostic of '
     'a small conditional inverse problem, not a 15-dimensional posterior validation.')
surrogate_path = REPORTS/'reference-posterior-surrogate-results.json'
if surrogate_path.exists():
    sr = read(surrogate_path.name)
    sa = read('reference-posterior-surrogate-final-independent-audit.json')
    if not sa.get('passed') or sa['evidence_sha256'].get(str(surrogate_path.relative_to(ROOT))) != EVIDENCE[str(surrogate_path.relative_to(ROOT))]:
        raise ValueError('Surrogate reference requires current independent audit')
    if not sa['qualified_developmental_surrogate_reference']:
        raise ValueError('Surrogate reference did not qualify')
    heading('Separate continuous numerical reference')
    endpoint = max(c.get('interval_endpoint_change_in_sd',0) for k,c in sr['checks'].items() if k!='original_moments')
    para('A separately planned repair fits a degree-four polynomial to cached forward-response vectors, with a '
         'degree-three crosscheck. It retains the original prior, likelihood and exact common-scale wrench law. '
         'All 64 fresh exact FEM validations pass, including samples from both proposal widths and an outer shell. '
         'Continuous joint scrambled-Sobol importance integration uses actual proposal-density correction and '
         'weighted empirical quantiles, rather than Gaussian intervals.')
    para(f'Across fixed sample levels, proposal widths, scrambles and polynomial degrees, the largest endpoint '
         f'change is {endpoint:.6f} posterior SD, below 0.05. Independent review recomputed all fresh response '
         'errors and the primary 1,048,576-sample integration. This qualifies the declared developmental numerical '
         'surrogate checks; it does not retroactively qualify the original Gauss-Hermite intervals.')
    summary = sr['primary_summary']
    table([['Log parameter','Mean','90% numerical interval']]+[[label,f'{mean:.6f}',f'[{ends[0]:.6f}, {ends[1]:.6f}]']
          for label,mean,ends in zip(['Bulk scale s','Left support log K','Right support log K'],summary['mean'],summary['interval90'])], [156,110,202])
    para('Tail checks are empirical: the primary proposal samples no radius-above-eight points, while wider '
         'proposals supply finite tail samples. Neither these estimates nor shell validation are a global error '
         'certificate. This is a three-parameter problem with known nuisance values, not exact-FEM posterior '
         'certification, population coverage or validation of the full 15-dimensional inference.', 'CaptionCustom')

page()
heading('6.4. Frozen-law transfer to normal contact')
contact_path = REPORTS/'contact-transfer-fine.json'
if contact_path.exists():
    contact = read(contact_path.name)
    ca = read('contact-transfer-final-independent-audit.json')
    if not ca.get('passed') or ca['evidence_sha256'].get(str(contact_path.relative_to(ROOT))) != EVIDENCE[str(contact_path.relative_to(ROOT))]:
        raise ValueError('Contact transfer requires current independent audit')
    para('A separate development experiment composes the frozen learned bulk and nonlinear bilateral '
         'attachment laws with a known analytic normal-penalty operator for a floor and sphere, under gravity. '
         'No network retraining is performed. Both material/interface combinations use known descriptors. '
         'All 16 analytic/shared equilibrium states completed; all four model-family mesh and contact-quadrature '
         'qualification sets passed, independently checked in 265 audit checks.')
    rows = [['Bulk + attachment', 'Field RMS (um)', 'Sphere force error (N)', 'Sphere force error (%)']]
    for family,label in [('mr_saturated','Mooney-Rivlin + saturating'),('yeoh_polynomial','Yeoh + polynomial')]:
        c = ca['transfer'][family]
        rows.append([label, f"{c['field_rms_um']:.3f}", f"{c['sphere_force_error_N']:.6f}", f"{100*c['sphere_force_relative_error']:.3f}"])
    table(rows,[180,96,96,96])
    para('Shared-law versus analytic-teacher differences on the finest n8/q32 states. These are descriptive '
         'transfer errors, not thresholds used to qualify the reference. Fixed-penalty mesh and quadrature '
         'consistency do not prove continuum convergence. Earlier coarse and tiny-contact failures remain preserved.', 'CaptionCustom')
    figure = REPORTS/'contact-transfer-fine-corrected.png'
    if figure.exists():
        EVIDENCE[str(figure.relative_to(ROOT))] = hashlib.sha256(figure.read_bytes()).hexdigest()
        from PIL import Image as PILImage
        with PILImage.open(figure) as im:
            width,height=im.size
        scale=min(468/width,245/height)
        story.append(Image(str(figure),width=width*scale,height=height*scale))
    para('Known geometry, oracle descriptors and static equilibrium only. Contact is normal and frictionless; '
         'rigid obstacle poses are prescribed. Finite penalty penetration remains. This does not establish '
         'unknown-contact inference, dynamic impact, soft-soft interaction or public-dataset improvement.')
else:
    para('Learned-law contact transfer has not supplied completed audited evidence.')

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
para('The six-object inverse study now provides evidence of observation-based refinement under hidden continuous '
     'material and support descriptors. Its fixed-action ties, incomplete fits and overconfident uncorrected bands '
     'remain part of the result. The completed oracle forward study includes both stabilized direct budgets and '
     'explicitly preserved runtime/preprocessing failures. Secondary reaction replay confirms that position-only '
     'refinement also improves force predictions, while fixed twist is slightly better for reactions than the '
     'position-targeted selected stretch. Two known-descriptor normal-contact compositions pass fixed-penalty '
     'numerical checks without retraining. The original VF reference remains unresolved at its ratio quadrature; '
     'a separately validated continuous numerical surrogate passes finite consistency checks with empirical, '
     'not globally certified, tail evidence. Existing repository benchmark results remain preserved '
     'but cannot be attributed to this new representation without adapting and rerunning it.')
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
                 'gravity-inference-pilot-results.json', 'scaling-identity-ablation-results.json',
                 'stable-zero-control-plan.json', 'stable-zero-prefit-independent-review.json',
                 'stable-zero-score-guard-independent-review.json', 'stable-zero-io-recovery-plan.json',
                 'stable-zero-io-recovery-prefit-independent-review.json',
                 'stable-zero-io-recovery-prediction-audit.json', 'stable-zero-io-direct-memory-check.json',
                 'stable-zero-io-reporting-amendment.json', 'oracle-missing-recovery-plan.json',
                 'oracle-recovery-original-diagnosis.json'):
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
