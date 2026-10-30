"""Additive report-only appendix; refuses incomplete or stale audited evidence.

Does not simulate, change selection, overwrite parent reports, or update summaries.
Run only after completed audit, fixed videos and independent source review.
"""
from pathlib import Path
import hashlib,json,sys
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.build_surface_report import p,table,fmt,STYLES
from scripts.coupled_contact_v13.fine_grid_report import summarize,parameter_diagnostics
from reportlab.platypus import SimpleDocTemplate,PageBreak,Paragraph,Spacer
from reportlab.lib.pagesizes import A4
REPORT=ROOT/'reports/coupled-friction-v13';PARENT=ROOT/'reports/coupled-surface-v13'
BASE=ROOT/'runs/coupled-friction-v13';OLD=ROOT/'runs/coupled-surface-v13'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def validate():
 protocol=read(REPORT/'protocol.json');freeze=read(REPORT/'method-freeze.json');summary=read(REPORT/'summary.json');audit=read(REPORT/'independent-audit.json');inputs={}
 def bind(path,expected=None):
  digest=sha(path)
  if expected is not None and digest!=expected:raise RuntimeError('Changed evidence: '+str(path))
  inputs[str(path.relative_to(ROOT))]=digest
 if audit.get('passed') is not True or audit.get('errors') or audit.get('pending'):raise RuntimeError('Passed completed independent audit required')
 bind(ROOT/'scripts/coupled_contact_v13/fine_grid_verify.py',audit['verifier_sha256'])
 for path,expected in audit['evidence_sha256'].items():bind(Path(path),expected)
 for name in ('protocol.json','method-freeze.json','development-results.json','reserved-results.json'):
  path=REPORT/name
  if str(path) not in audit['evidence_sha256']:raise RuntimeError('Audit lacks '+name)
 bind(REPORT/'protocol.json',freeze['protocol_sha256'])
 for path,expected in freeze['source_sha256'].items():bind(ROOT/path,expected)
 for name,key in [('protocol.json','parent_protocol_sha256'),('method-freeze.json','parent_method_freeze_sha256')]:
  bind(PARENT/name,freeze[key])
 if summary['protocol_sha256']!=sha(REPORT/'protocol.json') or summary['independent_audit'].get('sha256')!=sha(REPORT/'independent-audit.json') or summary['independent_audit'].get('passed') is not True:raise RuntimeError('Summary audit/protocol stale')
 for path,expected in summary['source_hashes'].items():bind(ROOT/path,expected)
 fine={};coarse={};selections={'fine':{},'coarse':{}}
 for grid,folder,lookup in [('fine',REPORT,fine),('coarse',PARENT,coarse)]:
  for split in ('development','reserved'):
   path=folder/f'{split}-results.json';document=read(path);bind(path)
   if summary['source_hashes'].get(str(path.relative_to(ROOT)))!=sha(path):raise RuntimeError('Summary missing result source')
   selections[grid].update(document['selections'])
   for row in document['cases']:
    key=(row['object'],row['scenario'])
    if key in lookup or row['split']!=split:raise RuntimeError('Duplicate or wrong-split case')
    lookup[key]=row
 expected={(o['id'],s) for o in protocol['objects'] for s in protocol['forecast_scenarios']}
 if len(expected)!=40 or set(fine)!=expected or set(coarse)!=expected:raise RuntimeError('All40 fine and coarse rows required')
 for split in ('development','reserved'):
  for cohort,scenes in [('primary',protocol['primary_forecast_scenarios']),('stress',protocol['stress_forecast_scenarios']),('full',protocol['forecast_scenarios'])]:
   keys=[(o['id'],s) for o in protocol['objects'] if o['split']==split for s in scenes]
   computed=summarize(keys,fine,coarse);computed['scenarios']=scenes
   if computed!=summary['cohorts'][split][cohort]:raise RuntimeError('Summary metric mismatch')
 if summary['development_primary_regressions']!=summary['cohorts']['development']['primary']['paired_comparisons']['coarse_fem']:raise RuntimeError('Development regression summary mismatch')
 if summary['cohorts']['reserved']['primary']['expected']!=24 or summary['cohorts']['reserved']['stress']['expected']!=8:raise RuntimeError('Fixed reserved cohort counts changed')
 if parameter_diagnostics(protocol,selections)!=summary['parameter_diagnostics']:raise RuntimeError('Parameter summary mismatch')
 if summary['library_coverage']!=audit['library']:raise RuntimeError('Library coverage mismatch')
 videos=read(REPORT/'video-provenance.json');expected_videos={f'{o}-{s}.mp4':(o,s) for o,s in protocol['video_cases']}
 if len(videos)!=3 or len(expected_videos)!=3 or {Path(v['path']).name for v in videos}!=set(expected_videos):raise RuntimeError('Fixed three videos required')
 for v in videos:
  rel=Path(v['path'])
  if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe video path')
  bind(ROOT/rel,v['sha256']);obj,scene=expected_videos[rel.name]
  for path,key in [(OLD/'reference'/obj/scene/'trajectory.npz','reference_sha256'),(OLD/'reference'/obj/scene/'complete.json','primary_status_sha256'),(BASE/'evaluation'/obj/scene/'predictions.npz','prediction_sha256')]:bind(path,v[key])
  for name,key in [('protocol.json','secondary_protocol_sha256'),('method-freeze.json','secondary_method_freeze_sha256'),('independent-audit.json','independent_audit_sha256')]:bind(REPORT/name,v[key])
  bind(ROOT/'scripts/coupled_contact_v13/fine_grid_render.py',v['fine_grid_renderer_sha256'])
  stress=scene in protocol['stress_forecast_scenarios']
  if v['frames']!=61 or v['simulation_fps']!=100 or v['playback_fps']!=25 or v['forecast_cohort']!=('stress' if stress else 'primary') or v['development_scope_qualified'] is not (not stress):raise RuntimeError('Video timing/cohort mismatch')
  if stress and 'unqualified' not in v.get('reference_display_label','').lower():raise RuntimeError('Stress video must be visibly unqualified')
 for name in ('summary.json','independent-audit.json','video-provenance.json'):bind(REPORT/name)
 bind(Path(__file__));bind(ROOT/'scripts/coupled_contact_v13/build_surface_report.py')
 return protocol,summary,videos,inputs

def footer(canvas,document):
 canvas.saveState();canvas.setFont('Helvetica',8);canvas.drawString(42,25,'Coupled friction v13 | Secondary simulation appendix');canvas.drawRightString(553,25,str(document.page));canvas.restoreState()

def main():
 protocol,s,videos,inputs=validate();primary=s['cohorts']['reserved']['primary'];stress=s['cohorts']['reserved']['stress']
 story=[p('Friction resolution and deformation prediction','Heading1'),p('Secondary study appendix | Position-only calibration, unchanged coupled physics','Heading2')]
 story+=[p('Motivation and controlled change','Heading2'),p('A development object selected excessive stiffness when its friction lay between coarse candidates. A uniform nested friction grid tests whether this discretization causes compensating material estimates. This diagnosis motivated a secondary study; it does not establish that friction resolution is the only source of error.'),p('The physical solver, predictor and reference meshes, timestep, contact, material candidates, observed positions and masks, noise realization, full-window calibration loss, cohort definitions and fixed videos are unchanged. No force observation is added. Reference and observation artifacts are reused read-only. The original coarse study is preserved.')]
 story.append(table([['Calibration family','Coarse candidates','Fine candidates'],['Nonlinear FEM','35 = 7 x 5','119 = 7 x 17'],['Adapted native spring core','35 = 7 x 5','119 = 7 x 17'],['Rigid contact baseline','5','17']],[245,115,115]))
 story+=[p('The fine friction grid spans 0 to 0.8 in steps of 0.05. Of 255 candidates, 75 reuse the exact coarse outputs and 180 are additional. Greater search budget is part of this comparison; it is not an equal-compute architecture experiment.'),p('Fixed reporting scope','Heading2'),p('All 40 development and reserved forecasts remain recorded, including failures. The reserved primary cohort contains all 24 slide, curved-obstacle and moving-body cases. Eight sharp-edge drops remain exploratory stress cases because development spatial qualification failed. Numerical qualification flags do not remove cases from either cohort. Means are conditional on completed predictions when coverage is incomplete.')]
 story.append(PageBreak());story.append(p('Reserved prediction results','Heading1'))
 rows=[['Method / search','Complete /24','World mm','Shape mm']]
 for key,label in [('coarse_fem','FEM, coarse grid'),('fine_fem','FEM, fine grid'),('fine_native','Adapted spring core, fine'),('fine_rigid','Rigid contact, fine'),('fine_oracle','True-parameter coarse FEM'),('fine_position_ridge','Position ridge')]:
  r=primary['methods'][key];rows.append([label,f"{r['complete']}/{r['expected']}",fmt(r['coordinate_mean_mm']),fmt(r['shape_mean_mm'])])
 story.append(table(rows,[235,80,80,80]));story.append(p('World values are coordinate RMSE; shape values remove best-fitting rigid motion. They measure simulation forecast error, not calibrated physical uncertainty. The oracle uses true parameters in the coarse predictor, so its residual includes discretization error.','Small'))
 rows=[['Fine FEM versus','Matched /24','Mean gain mm','Wins / losses / ties']]
 for key,label in [('coarse_fem','Coarse FEM'),('fine_rigid','Fine rigid'),('fine_native','Fine spring core'),('fine_oracle','Oracle')]:
  r=primary['paired_comparisons'][key];rows.append([label,f"{r['matched']}/{r['expected']}",fmt(r['gain_mm']),f"{r['wins']} / {r['losses']} / {r['ties']}"])
 story.append(table(rows,[170,90,90,125]));story.append(p('Positive paired gain favors fine FEM. Comparisons use only jointly completed cases, with matched counts shown; failures are not imputed as favorable errors. No significance or population generalization is inferred from these descriptive means.','Small'))
 r=stress['methods']['fine_fem'];q=stress['paired_comparisons']['coarse_fem'];d=s['development_primary_regressions']
 story+=[p('Stress and development accounting','Heading2'),p(f"Sharp-edge stress: fine FEM completes {r['complete']}/{r['expected']}; conditional world / shape means {fmt(r['coordinate_mean_mm'])} / {fmt(r['shape_mean_mm'])} mm. Fine-versus-coarse paired gain is {fmt(q['gain_mm'])} mm over {q['matched']}/{q['expected']} cases. These targets remain development-unqualified."),p(f"Development primary regression check: {d['matched']}/{d['expected']} matched cases; mean fine-grid gain {fmt(d['gain_mm'])} mm; {d['wins']} improvements, {d['losses']} regressions and {d['ties']} ties. Development cases motivated this secondary study and are not independent confirmation.")]
 story.append(PageBreak());story.append(p('Physical estimates, coverage and evidence','Heading1'))
 rows=[['Object / split','True E / mu','Coarse FEM E / mu','Fine FEM E / mu']]
 for split in ('development','reserved'):
  for r in s['parameter_diagnostics']['splits'][split]['objects']:
   def fit(key):
    f=r['fits'][key];return 'unavailable' if not f['success'] else f"{f['selected_young_pa']:.0f} / {f['selected_friction']:.2f}"
   rows.append([r['object']+' / '+('dev' if split=='development' else 'reserved'),f"{r['truth_young_pa']:.0f} / {r['truth_friction']:.2f}",fit('coarse_fem'),fit('fine_fem')])
 story.append(table(rows,[125,110,120,120]));story.append(p('E is in Pa. Truth appears only in post-fit diagnostics, not candidate selection. Point estimates do not demonstrate identifiability or calibrated uncertainty. Native spring scale is not Young modulus.','Small'))
 coverage=s['library_coverage'];story.append(p('Completed calibration candidates: '+', '.join(f"{name} {coverage[name]['complete']}/{coverage[name]['expected']}" for name in ('fem','native','rigid'))+'. All guarded failures remain in the independent audit. The native comparison adapts the authors\' spring-force core into shared contact and integration; it is not a full PhysTwin system reproduction. There is no matched fine-grid vertex-contact baseline.'))
 story.append(p('Fixed actual-versus-predicted videos','Heading2'))
 for v in videos:
  label=Path(v['path']).name+' | '+v['forecast_cohort']
  story.append(Paragraph('<link href="../../'+escape(v['path'],{'"':'&quot;'})+'" color="#176b8a">'+escape(label)+'</link>',STYLES['Small']))
 story.append(p('The same three predeclared cases include the sharp-edge stress example. Playback is quarter speed with shared scale and camera. A clip is illustrative and does not replace full-cohort accounting. This secondary simulation study makes no official benchmark improvement, calibrated boundary-inference, active-probing advantage or general contact-solver convergence claim.','Small'))
 out=ROOT/'papers/coupled-friction-v13';out.mkdir(parents=True,exist_ok=True);pdf=out/'report.pdf'
 SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=40,title='Coupled friction v13: secondary simulation appendix',author='PhysGauss-SDE research campaign').build(story,onFirstPage=footer,onLaterPages=footer)
 (REPORT/'paper-build.json').write_text(json.dumps({'pdf':str(pdf.relative_to(ROOT)),'pdf_sha256':sha(pdf),'input_sha256':inputs,'scope':'Additive reporting-only secondary appendix; source frozen science unchanged','visual_review':'pending'},indent=2)+'\n')
if __name__=='__main__':main()
