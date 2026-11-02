"""Independent merged recovery audit, retaining original bounded failures."""
import json,time,itertools
import numpy as np
from scripts.compositional_neural_mechanics_reporting.oracle_independent_audit import ROOT,R,OUT,Audit,read,sha,rest_points,metrics,bootstrap
B=ROOT/'runs/compositional-neural-mechanics-v16/oracle-missing-recovery'
def main():
 a=Audit();planpath=R/'oracle-missing-recovery-plan.json';p=read(planpath);oldpath=R/'oracle-reserved-forward-results.json';old=read(oldpath);fp=R/'oracle-method-freeze.json';f=read(fp);recordsfile=OUT/'reserved-records.json';records=read(recordsfile)['records'];resultpath=R/'oracle-recovery-merged-results.json';d=read(resultpath);indexpath=R/'oracle-recovery-reference-index.json';index=read(indexpath);terminal=read(B/'terminal.json')
 for path in (planpath,oldpath,fp,recordsfile,resultpath,indexpath,B/'terminal.json'):a.bind(path)
 for path,h in p['source_sha256'].items():a.check(sha(ROOT/path)==h,'recovery source')
 for path,h in f['source_sha256'].items():a.check(sha(ROOT/path)==h,'frozen source')
 for c in f['checkpoints'].values():a.check(sha(ROOT/c['path'])==c['sha256'],'checkpoint')
 a.check(p['original_result_sha256']==sha(oldpath)==d['original_result_sha256'],'original failures preserved');a.check(p['parent_freeze_sha256']==sha(fp)and p['records_sha256']==sha(recordsfile),'parent seals');a.check(d['recovery_plan_sha256']==index['plan_sha256']==terminal['plan_sha256']==sha(planpath),'recovery plan');a.check(d['reference_index_sha256']==sha(indexpath),'reference index')
 key=lambda x:(x['scene_id'],x['action'],x['method']);oldrows={key(x):x for x in old['rows']};missing={key(x)for x in old['rows']if x['state']!='complete'};reused={key(x)for x in p['reused']}
 a.check(len(missing)==48 and {key(x)for x in p['jobs']}==missing and len(p['jobs'])==48,'missing only');a.check(len(reused)==len(p['reused'])==240 and reused==set(oldrows)-missing,'exact reuse');a.check(len(terminal['jobs'])==48 and {x['index']for x in terminal['jobs']}==set(range(48)),'48terminal')
 expected={(r['scene']['name'],act['name'],m):(r,act)for r in records for act in r['actions']for m in ('reference',*f['methods'])};rows={key(x):x for x in d['rows']};a.check(len(rows)==len(d['rows'])==288 and set(rows)==set(expected),'288fixed');refs={key(x):x for x in index['rows']};a.check(len(refs)==len(index['rows'])==48 and set(refs)=={k for k in expected if k[2]=='reference'},'48reference index')
 arrays={};scores={};q=np.array(f['material_query'])
 for k,row in rows.items():
  a.check(row['cohort']==expected[k][0]['role']and row['state']in ('complete','failed'),'identity status')
  if k in reused:a.check(row['artifact']==oldrows[k]['artifact']and row['sha256']==oldrows[k]['sha256']and row['state']=='complete','unchanged reused')
  else:
   i=next(i for i,j in enumerate(p['jobs'])if key(j)==k);cp=B/f'job-{i:03d}/complete.json'
   if cp.exists():
    c=read(cp);a.bind(cp);a.check(c['plan_sha256']==sha(planpath)and key(c['row'])==k and c['row']['state']==row['state'],'new completion')
    if row['state']=='complete':a.check(c['row']['sha256']==row['sha256'],'completion array')
   else:a.check(row['state']=='failed','missing retained')
  if k[2]=='reference':a.check(refs[k]==row,'reference index exact')
  if row['state']!='complete':a.check('scores'not in row,'failed no score');continue
  path=ROOT/row['artifact'];a.check(sha(path)==row['sha256'],'array hash');a.bind(path)
  with np.load(path,allow_pickle=False)as raw:
   a.check(set(raw.files)=={'material_query','displacement','wrenches'}and np.array_equal(raw['material_query'],q),'schema/query');v={n:raw[n].copy()for n in ('displacement','wrenches')}
  a.check(v['displacement'].shape==(len(q),3)and v['wrenches'].shape==(2,6)and all(np.isfinite(x).all()for x in v.values()),'finite');arrays[k]=v
 for k,row in rows.items():
  if k[2]=='reference':continue
  ref=(k[0],k[1],'reference')
  if k not in arrays or ref not in arrays:a.check('scores'not in row,'no invalid scores');continue
  scores[k]=metrics(arrays[k],arrays[ref],rest_points(q,expected[k][0]['scene']['geometry']))
  a.check(set(scores[k])==set(row['scores']),'metric set')
  for name,v in scores[k].items():a.equal(v,row['scores'][name],'raw metric')
 means={};a.check(len(d['summaries'])==10,'ten summaries')
 for s in d['summaries']:
  role,m=s['cohort'],s['method'];keys=[k for k in expected if expected[k][0]['role']==role and k[2]==m];valid=[k for k in keys if k in scores];a.check(len(keys)==24 and s['expected_commands']==24 and s['objects']==6 and s['scored_commands']==len(valid)and s['complete']==(len(valid)==24),'counts')
  if len(valid)!=24:a.check('object_mean_scores'not in s,'no partial means');continue
  names=[r['scene']['name']for r in records if r['role']==role];means[role,m]={metric:np.array([np.mean([scores[k][metric]for k in keys if k[0]==name])for name in names])for metric in scores[keys[0]]}
  for metric,x in means[role,m].items():a.equal(x.mean(),s['object_mean_scores'][metric],'object means')
 expectedpairs={(role,m,metric)for role,m in means if m!='shared_laws'and(role,'shared_laws')in means for metric in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm')};a.check(len(d['paired_object_comparisons'])==len(expectedpairs)and {(x['cohort'],x['baseline'],x['metric'])for x in d['paired_object_comparisons']}==expectedpairs,'all pairs')
 for x in d['paired_object_comparisons']:
  v=bootstrap(means[x['cohort'],'shared_laws'][x['metric']],means[x['cohort'],x['baseline']][x['metric']])
  for k,value in v.items():a.check(x[k]==value,'bootstrap '+k)if isinstance(value,str)else a.equal(x[k],value,'bootstrap '+k)
 a.check(d['failed_predictions']==sum(x['state']!='complete'for x in rows.values()),'failures');a.check(d['recovered_rows']==sum(rows[k]['state']=='complete'for k in missing),'recovery count')
 output=dict(passed=True,errors=[],checks=a.checks,created_unix=time.time(),evidence_sha256=a.evidence,expected_rows=288,scored_rows=len(scores),original_missing_rows=48,recovered_rows=d['recovered_rows'],failed_predictions=d['failed_predictions'],scope='Identical frozen methods with separately budgeted missing-call recovery. Original bounded48failures preserved. Independent raw metrics and6-object bootstrap per cohort; n8 targets not individually mesh-qualified.')
 (R/'oracle-recovery-final-independent-audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items()if k!='evidence_sha256'}))
if __name__=='__main__':main()
