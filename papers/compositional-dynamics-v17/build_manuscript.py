"""Evidence-linked dynamic-mechanics supplement; never reads raw held targets.
Use runs/phystwin-env/bin/python. Final claims require the completed result reports.
"""
import hashlib,json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';OUT=ROOT/'output/pdf/compositional-dynamics-v17.pdf';EVIDENCE={};story=[]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('TitleV17',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#15394c'),spaceAfter=14))
styles.add(ParagraphStyle('BodyV17',parent=styles['BodyText'],fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle('CaptionV17',parent=styles['BodyText'],fontSize=8.5,leading=11,spaceAfter=8,textColor=colors.HexColor('#425466')))
styles.add(ParagraphStyle('CellV17',parent=styles['BodyText'],fontSize=8.3,leading=11))
styles.add(ParagraphStyle('EquationV17',parent=styles['BodyText'],fontName='Courier',fontSize=9,leading=13,backColor=colors.HexColor('#eff4f7'),borderPadding=8,spaceBefore=8,spaceAfter=15))
def read(name,required=True):
 p=R/name
 if not p.exists():
  if required:raise FileNotFoundError(p)
  return None
 EVIDENCE[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def text(s,style='BodyV17'):story.append(Paragraph(s,styles[style]))
def title(s):text(s,'Heading1')
def table(rows,widths):
 t=Table([[Paragraph(escape(str(c)),styles['CellV17'])for c in row]for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#668394')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
 story.extend([t,Spacer(1,9)])
def page():story.append(PageBreak())
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#425466'));canvas.drawString(.65*inch,.35*inch,'Compositional neural mechanics - dynamic simulation supplement');canvas.drawRightString(7.85*inch,.35*inch,str(doc.page))
protocol=read('fresh-cohort-protocol.json');native=read('native-budget-calibration-comparison.json');read('fine-refinement-final-independent-audit.json');read('cycle-refinement-final-independent-audit.json')
read('stronger-development-final-independent-audit.json');read('stronger-deformation-census.json');read('position-cycle-analysis.json')
development=read('matched-forecast-development-results.json',False)
cohort=read('fresh-cohort-geometry-recovery-results.json',False)or read('fresh-cohort-geometry-results.json',False)
appearance=read('matched-appearance-inference-recovery/dynamic-results.json',False)
damping=read('native-zero-damping-results.json',False)
cohort_images=read('fresh-cohort-appearance-results.json',False)
paired_images=read('fresh-cohort-appearance-paired-summary.json',False)
sensitivity=read('fresh-cohort-reference-sensitivity.json',False)
robustness=read('fresh-cohort-reference-robustness-posthoc.json',False)
if development:read('matched-forecast-development-final-independent-audit.json')
if appearance:read('matched-appearance-development-final-independent-audit.json')
if damping:read('native-zero-damping-final-independent-audit.json')
if cohort:
 read('fresh-cohort-geometry-final-independent-audit.json')
 read('fresh-cohort-shared-final-independent-audit.json')
 read('native-cohort-recovery-prediction-audit.json')
if cohort_images:read('fresh-cohort-appearance-final-independent-audit.json')
if sensitivity:read('fresh-cohort-reference-sensitivity-final-independent-audit.json')
if robustness:read('fresh-cohort-reference-robustness-final-independent-audit.json')
text('Physical accuracy beyond<br/>plausible dynamic rendering','TitleV17')
text('Research supplement | Simulation study | 2 November 2026','CaptionV17')
title('Research question')
text('A model can reconstruct observed motion and render an attractive scene while predicting the wrong response to a new load. We test these two abilities separately: future physical response under larger drops, and image quality from observed and new camera positions. Better physical prediction with comparable or slightly worse rendering is the target outcome, not a premise of the evaluation.')
if cohort and paired_images:
 ours=cohort['summary']['ours']['conditional_paired_complete_case_mean'];baseline=cohort['summary']['native_phystwin']['conditional_paired_complete_case_mean'];images=paired_images['means']
 text(f"On six paired forecasts across three fresh worlds, mean motion error is {1000*ours['world_point_rms_m']:.2f} mm versus {1000*baseline['world_point_rms_m']:.2f} mm for native PhysTwin; deformation error is {1000*ours['shape_point_rms_m']:.2f} versus {1000*baseline['shape_point_rms_m']:.2f} mm. Held-view foreground PSNR is {images['shared_q2']['psnr_foreground_union_db']:.2f} versus {images['native']['psnr_foreground_union_db']:.2f} dB. Our forecasts complete eight of eight cases; native completes six. These conditional results use the same initial appearance and finite-resolution synthetic references.")
title('Representation and prediction')
text('The representation separates material response, interactions and appearance. A shared neural bulk-energy model maps deformation and material descriptors to strain energy. Spatial material points move through a consistent-mass finite-element dynamic solver. Contact geometry supplies normal contact forces through a finite compliant law; prescribed tools supply position boundary conditions. A Gaussian appearance model is attached to material coordinates and transported with the deformation gradient.')
text('U(x,z) = integral W(F(x), z) dV + U_contact + U_tools<br/>M x_ddot = -dU/dx + M g<br/>mu_t = x_t(q) + F_t(q) offset<br/>Sigma_t = F_t(q) Sigma_0 F_t(q)^T','EquationV17')
text('The common experiment observes a short 10 mm drop. Its inputs are known initial geometry, total mass, gravity, and visible 3D surface positions; neither method receives the material family, its parameters, the finite contact stiffness, or future motion. Our bounded inverse search selects family, scale, shape and contact compliance using only those observations. The fitted model is then reset and predicts a higher drop without refitting.')
text('This study uses noiseless oracle 3D tracks with raster visibility, not end-to-end reconstruction from raw video. The global constitutive network was pretrained in earlier material-law studies; that prior is distinct from per-scene calibration. This extension does not establish calibrated posterior uncertainty or an active-probing advantage.','CaptionV17')
page();title('Visible deformation and actual dynamics')
picture=R/'videos/position-compression-development.png'
if picture.exists():
 EVIDENCE[str(picture.relative_to(ROOT))]=hashlib.sha256(picture.read_bytes()).hexdigest();story.append(Image(str(picture),width=6.9*inch,height=6.9*inch*1184/1536))
text('Recorded compression, hold and unload states. The prescribed tools close by 30% of the rest length; the material shortens by 22.7% because the attachments are compliant. The block bulges and recovers. Support crosses are tool annotations, not additional collision geometry. The video caption predates the final numerical audit; the smooth-cycle qualification below is now complete.','CaptionV17')
table([['Example','Observed behavior','Evidence scope'],['Position-controlled cycle','22.7% material shortening; recovery','Six mesh/time/quadrature states pass'],['Finite 80 g striker at 1 m/s','Local bending and rebound; about 13.6% central compression','Coupled dynamic demonstration; not spatially qualified'],['40 mm viscous drop, 0.75 s','Impact, rebound and settling','Separate dissipative scenario; refinement reported separately']], [1.65*inch,2.5*inch,2.75*inch])
text('Videos show saved integrator states at physical scale. Slow motion is labeled; displacement is not magnified. Striker and compression demonstrations are not presented as learned-prediction benchmark wins.','CaptionV17')
page();title('Numerical evidence and framework controls')
text('Refinement errors below use coordinate RMS. Forecast tables use vector RMS, which is sqrt(3) times coordinate RMS for the same weights.', 'CaptionV17')
table([['Reference check','Measured difference','Conclusion'],['Smooth compression n3 to n4','53.2 um world/shape coordinate RMS','Pass for this controlled cycle'],['20 mm conservative drop n6 to n8','143.45 um world; 141.66 um shape','Full declared refinement conjunction passes'],['Abrupt 20% release n6 to n8','211.08 um shape; 150 um gate','Unqualified; failure retained'],['Stronger abrupt release/drop exploration','Four guard failures; finite striker completes','No relaxed guards or replacement cases']], [2.35*inch,2.4*inch,2.15*inch])
text('Small energy drift alone does not qualify a simulation. The drop and controlled-cycle checks compare time step, spatial resolution, quadrature and orientation preservation; contact-event checks additionally apply to the drop. The controlled cycle has no floor impact. The qualified 20 mm stable Neo-Hookean drop does not automatically qualify the new MR/Yeoh materials or higher impacts. The fresh study therefore reports finite-resolution teacher sensitivity explicitly.')
text('Native Newton 1.7 / Warp 1.18 controls and the actual PhysTwin author simulator were executed. Newton development runs expose unresolved mesh/time and numerical-dissipation effects and are not used as a superior truth source. The main inferred-physics comparison uses author PhysTwin, with unchanged force and integration functions, known SI masses and a preserved fitted spring graph.')
table([['Native calibration configuration','Coordinate RMSE (mm)'],*[[{'original125':'125 points, 20 CMA / 200 epochs','capacity729':'729 points, 20 CMA / 200 epochs','extended125':'125 points, 100 CMA / 1000 epochs'}[r['variant']],f"{r['coordinate_rmse_mm']:.5f}"]for r in native['rows']if r['complete']]], [4.7*inch,2.2*inch])
text('The final baseline is globally selected from calibration replay only: 125 points, at most 100 CMA generations and 1000 gradient epochs. The 729-point capacity/configuration control and original 20/200 run remain reported. Logged online training loss is not substituted for fixed-checkpoint replay error. No further recipe selection uses the fresh targets.')
text('Both methods receive the same total mass, but native uniform particle masses and continuum consistent mass differ. The 125-point grid has 1.5 times the continuum rest rotational inertia; 729 points have 1.25 times. Any prediction advantage is a configuration-level result that can jointly reflect representation, mass/contact discretization, material prior and inference.','CaptionV17')
page();title('Fresh prediction and rendering protocol')
text('Four new worlds use balanced Mooney-Rivlin and Yeoh material families, privately drawn stiffness/shape/contact settings, a fixed 12 x 6 x 6 cm block, 0.27 kg mass and gravity 9.8 m/s<super>2</super>. Each supplies 21 calibration frames over 0.20 s. The two forecasts are 40 mm and 80 mm drops, each 0.30 s with 31 common frames. This is load-magnitude extrapolation; it is not a new interaction-type or geometry benchmark.')
text('Primary references use n4 discretization. A prospectively selected subset receives n6 and half-step sensitivity checks. All original worlds, failed prefixes and unavailable predictions remain in the eight-forecast census. There are four independent objects, not eight. Primary paired means require all eight valid complete forecasts; otherwise coverage and explicitly conditional complete-case summaries are shown.')
table([['Physical prediction','Rendering quality'],['Rest-area weighted world-position vector RMS','Foreground-union PSNR and SSIM'],['Shape RMS after proper rigid alignment only','Silhouette IoU'],['Nonrigid deformation and surface-centroid error','Joint-visible depth RMSE and coverage'],['Sampled penetration, closed-surface volume and bounce events','Observed-camera future frames and new camera views']], [3.45*inch,3.45*inch])
text('The appearance comparison starts both methods from the same initial learned Gaussian checkpoint, trained only on initial views. PhysTwin uses its native K16 binding; our method uses material-coordinate centers and F Sigma F^T covariance transport. The common renderer, opacity rule, cameras and backgrounds are fixed. Thus image differences reflect motion and transport jointly. A separate procedural rendering isolates geometry without claiming native appearance parity.')
text('All method/query physics attempts must be committed before target arrays are opened. All predicted image slots must be committed before reference images are generated. A 2 dB foreground PSNR or 0.02 SSIM difference is a descriptive practical band, not a statistical equivalence test. Physical and image scores remain separate; no composite score can hide a physically wrong response.')
if development or cohort:page()
title('Development: motion and rendering')
if development:
 rows=[['Development method','World RMS (mm)','Shape RMS (mm)']]
 labels={'ours_coarse':'Ours, coarse n2','ours_fine':'Ours, fine n3','original125':'Native 125 points, 20/200','capacity729':'Native 729 points, 20/200','extended125':'Native 125 points, 100/1000 (selected)'}
 for name,result in development['methods'].items():
  g=result['geometry'];rows.append([labels[name],f"{1000*g['world_point_rms_m']:.4f}"if g.get('complete')else'incomplete',f"{1000*g['shape_point_rms_m']:.4f}"if g.get('complete')else'incomplete'])
 table(rows,[3.3*inch,1.8*inch,1.8*inch])
 text('One development object, with a same-n3 finite-resolution teacher for the fine variant. This is not a population or independently mesh-qualified prediction result. All variants are retained; the primary native configuration was selected using calibration only.','CaptionV17')
 if appearance:
  rows=[['Prediction / views','Foreground PSNR (dB)','Foreground SSIM','IoU']]
  for method,label in [('shared_q2','Ours'),('native','Native PhysTwin')]:
   for role,view in [('train','observed views'),('held','new views')]:
    m=appearance['summary'][method][role]['metrics'];rows.append([label+' / '+view,f"{m['psnr_foreground_union_db']['mean']:.2f}",f"{m['ssim_rgb_foreground_union']['mean']:.3f}",f"{m['silhouette_iou']['mean']:.3f}"])
  table(rows,[2.85*inch,1.45*inch,1.45*inch,1.15*inch])
  text('Both methods use the same initial Gaussian appearance and native renderer. These scores measure motion plus transport, not independently reconstructed appearance. Frames/views are correlated; no population confidence interval is inferred from one object.','CaptionV17')
 if damping:
  d=damping['methods']['zero_drag_and_dashpot']['forecast']['geometry']
  text(f"A post-hoc mechanism diagnostic removes only the selected baseline's fitted drag and dashpot, without refitting. World error falls to {1000*d['world_point_rms_m']:.3f} mm, but shape error rises to {1000*d['shape_point_rms_m']:.3f} mm. Damping therefore explains part of the motion discrepancy, not the entire gap. This diagnostic does not replace the selected baseline or change the fresh cohort. Our elastic-only prior matches the teacher's zero physical damping; the comparison is not an isolated test of material-law superiority.")
else:text('The matched development forecasts are still being completed. No future-motion superiority or native rendering-parity claim is made at this checkpoint.')
if cohort:
 page();title('Fresh cohort: physical prediction')
 rows=[['World / drop','Ours world mm','Native world mm','Ours shape mm','Native shape mm']]
 for case in cohort['rows']:
  values=[]
  for metric in ('world_point_rms_m','shape_point_rms_m'):
   for method in ('ours','native_phystwin'):
    result=case['methods'].get(method,{})
    value=result.get('geometry',{}).get(metric)if case['reference_complete']and result.get('completed')else None
    values.append(f'{1000*value:.3f}'if value is not None else'unavailable')
  rows.append([case['world_id']+' / '+case['role'].replace('forecast-',''),*values])
 table(rows,[2.1*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
 rows=[['Fresh method','Completed','Paired coverage','All-eight paired world RMS(mm)']]
 for name,result in cohort['summary'].items():
  value=result['primary_all_eight_mean'];complete=sum(bool(c['methods'].get(name,{}).get('completed'))for c in cohort['rows']);rows.append(['Ours'if name=='ours'else'Native PhysTwin',f'{complete}/8',f"{result['paired_complete_forecasts']}/8",f"{1000*value['world_point_rms_m']:.4f}"if value else'unavailable'])
 table(rows,[2.3*inch,1.3*inch,1.3*inch,2.0*inch])
 text('Unavailable is a failed or missing prediction/reference, never zero error. Each physical metric uses the complete saved sequence at matching times; only the shape metric removes proper rigid motion. The eight rows are two loads on each of four worlds, not eight independent objects.','CaptionV17')
 for name,label in [('ours','Ours'),('native_phystwin','Native PhysTwin')]:
  result=cohort['summary'][name];m=result['conditional_paired_complete_case_mean'];n=result['paired_complete_forecasts']
  if n:
   text(f"{label}: on the same {n}/8 available paired forecasts only, mean world error is {1000*m['world_point_rms_m']:.3f} mm and mean shape error is {1000*m['shape_point_rms_m']:.3f} mm. These are conditional summaries; they do not substitute for an all-eight result.")
 if sensitivity:
  spatial=[c for c in sensitivity['comparisons']if c['kind']=='spatial'];temporal=[c for c in sensitivity['comparisons']if c['kind']=='temporal']
  text(f"Reference sensitivity: {sensitivity['completed_jobs']}/20 declared simulations completed. Spatial checks passed {sum(c['passed']for c in spatial)}/{len(spatial)}; time-step checks passed {sum(c['passed']for c in temporal)}/{len(temporal)}. The subset checks the 80 mm drop at n4/n6 and both time steps, plus calibration spatial consistency. It does not provide a continuum solution.")
 text('Native world-3 fitting stopped during graph initialization, leaving both forecasts unavailable. Our four selected fits used the fixed calibration budget; only one of the sixteen coarse/fine family optimizations reported convergence. Completed forecasts and optimizer convergence are distinct outcomes.','CaptionV17')
else:text('Fresh cohort generation, inference and prediction are active. Final physical and rendering tables remain pending; the previous successful benchmark artifacts are preserved.')
if robustness:
 page();title('Reference-resolution robustness')
 text('After the declared spatial checks failed, a secondary diagnostic evaluates the same frozen predictions against the already generated finer references. It uses both prospectively fixed subset worlds and the 80 mm load. This is post-hoc analysis, with no refitting, new simulations or replacement of the primary n4 scores.')
 rows=[['World / reference','Ours world mm','Native world mm','Ours shape mm','Native shape mm']]
 for case in robustness['rows']:
  values=[]
  for metric in ('world_point_rms_m','shape_point_rms_m'):
   for method in ('ours','native_phystwin'):
    v=case['methods'][method].get(metric);values.append(f'{1000*v:.3f}'if v is not None else'unavailable')
  rows.append([case['world_id']+' / '+case['reference'].replace('_',' '),*values])
 table(rows,[2.1*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
 text('Absolute errors depend on reference resolution. These rows permit a direct check of whether the method ordering changes with finer space/time discretization. A stable ordering does not repair the failed spatial qualification or establish continuum accuracy. The experiment remains a comparison of matched-observation configurations on finite-resolution synthetic dynamics.')
if cohort_images:
 page();title('Fresh cohort: new-view image prediction')
 rows=[['World / method','Scored / expected','Foreground PSNR','Foreground SSIM','IoU']]
 for row in cohort_images['per_world_summary']:
  if row['camera_role']!='held':continue
  values=[]
  for metric in ('psnr_foreground_union_db','ssim_rgb_foreground_union','silhouette_iou'):
   m=row['metrics'][metric];v=m['full_coverage_mean'];values.append(f'{v:.3f}'if v is not None else'unavailable')
  label='Ours'if row['method']=='shared_q2'else'Native'
  rows.append([row['world_id']+' / '+label,f"{row['scored']}/{row['expected']}",*values])
 table(rows,[2.05*inch,1.2*inch,1.25*inch,1.25*inch,1.15*inch])
 text('Each row averages both forecast loads, all 31 times and the two fixed held cameras within one world. Full-coverage values are shown; missing images remain unavailable. Higher PSNR/SSIM/IoU is better. Per-frame values, observed-camera scores, depth error and coverage are retained in the complete image report.','CaptionV17')
 text('Both methods use the same initial Gaussian checkpoint and native rasterizer, with their respective motion/shape transport. This measures future-view prediction quality jointly with dynamics. It does not establish equality of independently learned appearance models. Correlated image frames are not treated as independent statistical samples.')
text('Reproduction: reports/compositional-dynamics-v17 contains frozen protocols, source/input hashes, audit records and NATIVE-SETUP.txt. Raw simulation datasets and model checkpoints remain local. This supplement extends the preserved v16 manuscript rather than replacing its results.','CaptionV17')
if development:
 page();title('Development response through impact and rebound')
 figure=R/'figures/development-physical-trajectories.png'
 if figure.exists():
  EVIDENCE[str(figure.relative_to(ROOT))]=hashlib.sha256(figure.read_bytes()).hexdigest();story.append(Image(str(figure),width=7.0*inch,height=7.0*inch*6.5/10))
 text('Top: unaligned world error and separately rigid-aligned shape error. Bottom: surface centroid (not mass center of mass) and the signed closed-surface volume ratio. The selected native model falls too slowly before impact and misses the rebound; the fine shared model tracks the reference closely. The coarse shared ablation retains a larger deformation error. All methods use the same recorded times; there is no temporal alignment.','CaptionV17')
 text('The sampled-volume diagnostic does not certify interior orientation or energy conservation. Reference and method-specific energy/Jacobian checks are reported separately. The damping intervention is explanatory and post-hoc; no further model tuning follows the development outcome. The fresh four-world study keeps the prospectively frozen recipe.')
 title('Sources and reproducibility')
 text('[1] Hanxiao Jiang et al. PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos. ICCV 2025. <link href="https://github.com/Jianghanxiao/PhysTwin" color="#0072B2">Author implementation</link>, pinned commit 81c718790a37e5e0102eb77af2c6edd34a9db25f.', 'CaptionV17')
 text('[2] Newton developers. <link href="https://newton-physics.github.io/newton/1.7.0/solvers/vbd.html" color="#0072B2">Newton 1.7 VBD solver documentation</link>. The executed runtime is pinned separately in this project.', 'CaptionV17')
 text('[3] Project protocols, raw-to-report hashes, independent audits and reproduction commands: reports/compositional-dynamics-v17. Earlier method and benchmark evidence: papers/compositional-neural-mechanics-v16. New raw datasets and trained checkpoints remain local.', 'CaptionV17')
OUT.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(OUT),pagesize=(8.5*inch,11*inch),leftMargin=.65*inch,rightMargin=.65*inch,topMargin=.55*inch,bottomMargin=.6*inch,title='Physical accuracy beyond plausible dynamic rendering',author='physgauss-sde research campaign').build(story,onFirstPage=footer,onLaterPages=footer)
meta=dict(output=str(OUT.relative_to(ROOT)),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),evidence_sha256=EVIDENCE,development_scored=development is not None,fresh_cohort_scored=cohort is not None,scope='Dynamic simulation supplement; matched-observation finite-resolution comparison, with coverage and reference sensitivity retained.')
Path(__file__).with_name('manuscript-evidence.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta|{'evidence_sha256':f'{len(EVIDENCE)}sources'}))
