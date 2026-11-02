"""Independent96-row stable normalization control audit; no production scorer."""
import json,time
import numpy as np
from scripts.compositional_neural_mechanics_reporting.oracle_independent_audit import ROOT,R,OUT,Audit,read,sha,metrics,rest_points,bootstrap
B=ROOT/'runs/compositional-neural-mechanics-v16/stable-zero-io-recovery'
def main():
 a=Audit(); parent_audit_path=R/'oracle-recovery-final-independent-audit.json';parent_audit=read(parent_audit_path);a.check(parent_audit.get('passed') is True,'primary independent audit required');a.bind(parent_audit_path)
 for path,h in parent_audit['evidence_sha256'].items():a.check(sha(ROOT/path)==h,'primary audit evidence')
 io_planpath=R/'stable-zero-io-recovery-plan.json';io_plan=read(io_planpath);a.bind(io_planpath);oldcommit=ROOT/'runs/compositional-neural-mechanics-v16/stable-zero-control/predictions-commit.json';a.check(sha(oldcommit)==io_plan['old_commit_sha256'],'old96failures preserved');a.bind(oldcommit)
 for path,h in io_plan['source_sha256'].items():a.check(sha(ROOT/path)==h,'I/O repair source')
 pred_audit_path=R/'stable-zero-io-recovery-prediction-audit.json';pred_audit=read(pred_audit_path);a.bind(pred_audit_path);a.check(pred_audit['passed'] and pred_audit['successful']==96 and pred_audit['finite']==96 and pred_audit['commit_sha256']==sha(B/'predictions-commit.json'),'96finite replay commitment')
 planpath=R/'stable-zero-control-plan.json';plan=read(planpath);fpath=R/'oracle-method-freeze.json';f=read(fpath);resultpath=R/'stable-zero-io-recovered-mixed-results.json';result=read(resultpath);cp=B/'predictions-commit.json';commit=read(cp);rp=OUT/'reserved-records.json';records=read(rp)['records']
 for p in (planpath,fpath,resultpath,cp,rp):a.bind(p)
 for table in ('source_sha256','input_sha256'):
  for p,h in plan.get(table,{}).items():a.check(sha(ROOT/p)==h,'control pin '+p)
 for p,h in f['source_sha256'].items():a.check(sha(ROOT/p)==h,'parent pin '+p)
 for value in plan['variants'].values():a.check(sha(ROOT/value['path'])==value['sha256'],'control checkpoint')
 a.check(commit['control_plan_sha256']==sha(planpath),'control plan');a.check(commit['parent_freeze_sha256']==sha(fpath)and commit['records_sha256']==sha(rp),'parent/record seals')
 expected={(r['scene']['name'],x['name'],m):(r,x)for r in records for x in r['actions']for m in plan['variants']}
 a.check(len(expected)==96,'expected96')
 rows={(x['scene_id'],x['action'],x['method']):x for x in result['rows']};committed={(x['scene_id'],x['action'],x['method']):x for x in commit['rows']}
 a.check(len(rows)==len(result['rows'])==len(committed)==len(commit['rows'])==96 and set(rows)==set(committed)==set(expected),'exact96')
 q=np.array(f['material_query']);scores={}
 for key,row in rows.items():
  record,_=expected[key];a.check(row['cohort']==record['role']and row['state']==committed[key]['state'],'identity/status')
  if row['state']!='complete':a.check(row['state']=='failed'and 'scores'not in row,'retained failure');continue
  pp=ROOT/row['artifact'];a.check(sha(pp)==row['sha256']==committed[key]['sha256'],'prediction seal');a.bind(pp)
  complete=R/'oracle-recovery-reference-index.json';a.bind(complete);rr=[x for x in read(complete)['rows']if x['scene_id']==key[0]and x['action']==key[1]and x['method']=='reference'];a.check(len(rr)==1,'reference identity')
  if rr[0]['state']!='complete':a.check('scores'not in row,'no score on failed reference');continue
  ref=ROOT/rr[0]['artifact'];a.check(sha(ref)==rr[0]['sha256'],'reference seal');a.bind(ref)
  data=[]
  for path in (pp,ref):
   with np.load(path,allow_pickle=False)as raw:
    a.check(np.array_equal(raw['material_query'],q),'query identity');v={k:raw[k].copy()for k in ('displacement','wrenches')}
   a.check(v['displacement'].shape==(len(q),3)and v['wrenches'].shape==(2,6)and all(np.isfinite(x).all()for x in v.values()),'finite data');data.append(v)
  values=metrics(*data,rest_points(q,record['scene']['geometry']));scores[key]=values;a.check(set(row['scores'])==set(values),'metric census')
  for k,v in values.items():a.equal(v,row['scores'][k],'independent '+k)
 means={};a.check(len(result['summaries'])==4 and {(x['cohort'],x['method'])for x in result['summaries']}=={(c,m)for c in ('reserved_composition','reserved_geometry')for m in plan['variants']},'four unique summaries')
 for s in result['summaries']:
  keys=[k for k,(r,_)in expected.items()if r['role']==s['cohort']and k[2]==s['method']];valid=[k for k in keys if k in scores]
  a.check(len(keys)==24 and s['expected_commands']==24 and s['scored_commands']==len(valid)and s['complete']==(len(valid)==24),'summary denominators')
  if len(valid)!=24:a.check('object_mean_scores'not in s,'no partial means');continue
  names=sorted({k[0]for k in keys});a.check(len(names)==6,'six object unit');means[s['cohort'],s['method']]={metric:np.array([np.mean([scores[k][metric]for k in keys if k[0]==name])for name in names])for metric in scores[keys[0]]}
  for metric,values in means[s['cohort'],s['method']].items():a.equal(values.mean(),s['object_mean_scores'][metric],'equal object mean')
 comparisons=[];primarypath=R/'oracle-recovery-merged-results.json';primary=read(primarypath);a.bind(primarypath)
 for (cohort,method),values in means.items():
  names=sorted({k[0]for k,(r,_)in expected.items()if r['role']==cohort});shared=[x for x in primary['rows']if x['method']=='shared_laws'and x['cohort']==cohort]
  if len(shared)==24 and all('scores'in x for x in shared):
   for metric in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm'):
    ss=np.array([np.mean([x['scores'][metric]for x in shared if x['scene_id']==name])for name in names]);comparisons.append(dict(cohort=cohort,baseline=method,metric=metric,**bootstrap(ss,values[metric])))
 combinedpath=R/'oracle-recovered-plus-stable-io-comparison.json';combined=read(combinedpath);a.bind(combinedpath);a.check(combined['merged_ledger_sha256']==sha(primarypath) and combined['stable_results_sha256']==sha(resultpath),'seven method table seals');a.check(len(combined['rows'])==14,'seven methods by two cohorts');expected_table=[dict(x,source='bounded_missing_call_recovery')for x in primary['summaries']]+[dict(x,source='supplementary_stable_zero_control')for x in result['summaries']];a.check(combined['rows']==expected_table,'exact seven method table')
 for p,h in result['evidence_sha256'].items():a.check(sha(ROOT/p)==h,'score evidence')
 a.check(len(scores)==96 and all(x['state']=='complete'for x in rows.values()),'usable baseline requires96finite scoredpredictions')
 output=dict(passed=True,baseline_comparison_complete=True,checks=a.checks,created_unix=time.time(),evidence_sha256=a.evidence,expected_predictions=96,scored_predictions=len(scores),failed_predictions=sum(x['state']!='complete'for x in rows.values()),summaries=result['summaries'],shared_laws_comparisons=comparisons,scope='Independent raw-array world/canonical/Kabsch/force/torque recomputation. All96 fixed forecasts retained; objectbootstrap6percohort. Primary shared-law inputs use separately audited primary scores. Directory-onlyrepair; weights/rule preoutcomes, firstsuccessfulforecastartifacts postoutcomes. Original96IOfailures preserved.')
 (R/'stable-zero-io-final-independent-audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'passed':True,'checks':a.checks,'scored':len(scores)}))
if __name__=='__main__':main()
