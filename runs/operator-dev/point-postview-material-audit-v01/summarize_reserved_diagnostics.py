"""All32 released-object descriptive materialerrors/ESS tails; no modelqueries."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from experiments import campaign_point_postview_material_audit as audit
from experiments.run_point_prior_analysis import validate_execution,eligible_objects
from experiments.summarize_point_prior_analysis import validate_freeze
ROOT=Path.cwd();OUT=ROOT/'runs/operator-dev/point-postview-material-audit-v01'
EXEC='runs/operator-dev/point-prior-analysis-v01/execution-protocol.json';ESHA='fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c'
RELEASE=dict(path='runs/operator-dev/point-prior-analysis-v01/reserved-release.json',sha256='ed47e4c535c0005efb978a148a7d5a4a3c0b6df32077621d14fad1640d0e9c28')
CALIB=dict(path='runs/gaussian-campaign-v01/point-prior-analysis-v01/calibration-frozen.json',sha256='80c61a592588cd61c571caf3804bcaefe5d0cea2627c656ae590f08655dac3df')
SEALED=dict(path='runs/gaussian-campaign-v01/point-prior-analysis-v01/reserved-evaluation.json',sha256='51fcbd0e93e92e2a6f80ab996c45c30b21e73c57c797cd436bca4549b733240d')
SUMMARY=dict(path='runs/gaussian-campaign-v01/point-postview-material-audit-v01/RESERVED_EVALUATION/summary.json',sha256='d04bfac0e510aed0796f2371984c2c14f7fd0d077b979fcafeb72128e37edc70')
LOCAL=ROOT/'runs/gaussian-campaign-v01/point-postview-material-audit-v01/RESERVED_EVALUATION/object-diagnostics-LOCAL.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
declaration=dict(scope='All32 alreadysealed/releasedobjects: savedpostview intervalmeanerror/supportconfidence/ESS tails only; no model/query/tuning',
 execution=dict(path=EXEC,sha256=ESHA),release=RELEASE,calibration=CALIB,sealed_primary_evaluation=SEALED,audit_summary=SUMMARY,audit_object_diagnostics=rec(LOCAL),
 source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/campaign_point_postview_material_audit.py':sha(ROOT/'experiments/campaign_point_postview_material_audit.py')},
 all_object_indices=list(range(16,48)),tail_objects_per_policy=5,metric_choices='Coverage/width,physicalunit MAE/RMSE/bias ofposterior mean,relativeMAE, ESSmin/p05/median/mean/max, supportprob/confidence/entropy and5lowestESS rows',
 statistical_role='Additional descriptive diagnostic aftersealedprimary evaluation; no new primaryendpoint or modelselection claim')
dump(OUT/'reserved-supplement-declaration.json',declaration)
execution,_,plan=validate_execution(EXEC,ESHA,check_environment=False);freeze=validate_freeze(CALIB,ESHA);release=audit._read(audit._checked(RELEASE))
if release['calibration_freeze']!=CALIB:raise ValueError('Exactcalibration freeze required')
objects=eligible_objects(plan,'RESERVED_EVALUATION',execution_sha256=ESHA,method_sha256=audit.METHOD_SHA,release=release,calibration_freeze=freeze,calibration_freeze_record=CALIB)
records,status=audit.preflight_saved_objects(objects,execution,ESHA,'RESERVED_EVALUATION');sealed=audit._read(audit._checked(SEALED));summary=audit._read(audit._checked(SUMMARY))
if not sealed['completed'] or sealed['objects']!=32 or not summary['completed'] or summary['role']!='RESERVED_EVALUATION' or summary['objects']!=32:raise ValueError('Complete32sealedobjects required')
rows=audit._read(audit._checked(declaration['audit_object_diagnostics']))
if [r['object_index'] for r in rows]!=list(range(16,48)):raise ValueError('No subset/failedcase filtering')
truth=[]
for row,record in zip(rows,records):
 if (record['result'] not in sealed['reserved_results'] or row['source_artifacts']['result']!=record['result']
     or row['selected_active_view']!=record['selected_active_view'] or row['selected_random_view']!=record['selected_random_view']
     or row['private_hypothesis']!=record['object']['realized_hypothesis']):raise ValueError('Audit/result/choice/privateprovenance mismatch')
 spec=audit._read(audit._checked(row['private_hypothesis']));truth.append([spec['material'][k] for k in audit.MATERIAL_KEYS])
truth=np.array(truth);best=freeze['best_fixed_view_index'];policies=['no_additional_view',*[f'fixed_view_{j}' for j in range(4)],'active','random','best_fixed_on_calibration'];result={}
base_means=np.array([[m['mean'] for m in row['diagnostic']['states']['no_additional_view']['material']] for row in rows]);base_errors=np.abs(base_means-truth)
for policy in policies:
 states=[]
 for row in rows:
  key=f"fixed_view_{row['selected_active_view']}" if policy=='active' else f"fixed_view_{row['selected_random_view']}" if policy=='random' else f'fixed_view_{best}' if policy=='best_fixed_on_calibration' else policy
  states.append(row['diagnostic']['states'][key])
 materials=[]
 for j,label in enumerate(audit.MATERIAL_NAMES):
  means=np.array([s['material'][j]['mean'] for s in states]);errors=means-truth[:,j];absolute=np.abs(errors);widths=np.array([s['material'][j]['interval_width'] for s in states]);hits=np.array([s['material'][j]['truth_in_90_percent_interval'] for s in states])
  for k,s in enumerate(states):
   m=s['material'][j]
   if bool(m['quantile05']<=truth[k,j]<=m['quantile95'])!=bool(hits[k]):raise ValueError('Suppliedtruth/intervalcoverage discrepancy')
  materials.append(dict(name=label,objects=32,covered=int(hits.sum()),coverage_fraction=float(hits.mean()),mean_interval_width=float(widths.mean()),
   posterior_mean_MAE=float(absolute.mean()),posterior_mean_RMSE=float(np.sqrt(np.mean(errors**2))),posterior_mean_bias=float(errors.mean()),
   posterior_mean_relative_MAE=float(np.mean(absolute/truth[:,j])),paired_mean_absolute_error_change_from_noaction=float(np.mean(absolute-base_errors[:,j])),
   objects_with_mean_error_improved=int(np.sum(absolute<base_errors[:,j])),objects_with_mean_error_worse=int(np.sum(absolute>base_errors[:,j]))))
 ess=np.array([s['empirical_ESS'] for s in states]);prob=np.array([s['support_true_node_probability'] for s in states]);confidence=np.array([s['support_mode_probability'] for s in states]);ranks=np.array([s['support_true_node_rank'] for s in states]);order=np.argsort(ess,kind='stable');wrong=ranks!=1
 result[policy]=dict(material=materials,empirical_row_ESS=dict(minimum=float(ess.min()),quantile05=float(np.quantile(ess,.05)),median=float(np.median(ess)),mean=float(ess.mean()),maximum=float(ess.max())),
  support_true_probability=dict(minimum=float(prob.min()),quantile05=float(np.quantile(prob,.05)),median=float(np.median(prob)),mean=float(prob.mean()),maximum=float(prob.max()),exactzero_objects=int(np.sum(prob==0))),
  support_mode_confidence=dict(mean=float(confidence.mean()),maximum=float(confidence.max()),wrong_mode_objects=int(wrong.sum()),mean_confidence_when_wrong=float(confidence[wrong].mean()) if wrong.any() else None),
  support_truth_rank1_objects=int((ranks==1).sum()),mean_support_entropy_nats=float(np.mean([s['support_entropy_nats'] for s in states])),
  positive_weight_geometry_ids=dict(minimum=min(s['positive_weight_geometry_ids'] for s in states),mean=float(np.mean([s['positive_weight_geometry_ids'] for s in states]))),
  common_empirical_geometry_ids=dict(minimum=min(s['unique_geometry_ids_in_common_particles'] for s in states),mean=float(np.mean([s['unique_geometry_ids_in_common_particles'] for s in states]))),
  five_lowest_ESS_objects=[dict(object_index=rows[int(i)]['object_index'],ESS=float(ess[i]),largest_weight=states[int(i)]['maximum_particle_weight'],true_support_probability=float(prob[i]),mode_support_probability=float(confidence[i]),true_support_rank=int(ranks[i]),positive_weight_geometry_ids=states[int(i)]['positive_weight_geometry_ids']) for i in order[:5]])
report=dict(completed=True,role='RESERVED_EVALUATION',objects=32,all_indices=list(range(16,48)),declaration=rec(OUT/'reserved-supplement-declaration.json'),original_stage_terminal=status,
 sealed_primary_evaluation=SEALED,base_audit_summary=SUMMARY,calibration_freeze=CALIB,reserved_release=RELEASE,policies=result,
 limitations=['Descriptiveadditionaldiagnostic aftersealedprimary evaluation; no model/selection tuning.','SAME2048resampled empiricalparticles before/after; rowESS not independenthypothesiscount; C28full56320reference intervals remainseparate.','Marginalempirical5/95parameterintervalcoverage is not conformalpredictionbandcoverage or Bayesianposterior guarantee.','All32cases retained, no new query/physics/retraction/model evaluations.'])
dump(OUT/'reserved-supplement-summary.json',report)
print(json.dumps({'output':rec(OUT/'reserved-supplement-summary.json'),'policies':{p:{'material':result[p]['material'],'ESS':result[p]['empirical_row_ESS'],'support':result[p]['support_true_probability'],'confidence':result[p]['support_mode_confidence']} for p in ['no_additional_view','active','best_fixed_on_calibration','random']}}))
