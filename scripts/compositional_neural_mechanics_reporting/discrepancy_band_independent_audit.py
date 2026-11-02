import json,hashlib,time
from pathlib import Path
import numpy as np
root=Path.cwd();r=root/'reports/compositional-neural-mechanics-v16';b=root/'runs/compositional-neural-mechanics-v16/unknown-action-reserved';d=json.load(open(r/'unknown-action-discrepancy-band-results.json'));a=json.load(open(r/'unknown-action-final-independent-audit.json'));v=json.load(open(r/'model-discrepancy-variance-freeze.json'));h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();errors=[];checks=0

def ck(ok,msg):
 global checks
 checks+=1
 if not ok:errors.append(msg)
ck(a['passed'],'primary passed');ck(d['primary_audit_sha256']==h(r/'unknown-action-final-independent-audit.json'),'audit seal');ck(d['variance_sha256']==h(r/'model-discrepancy-variance-freeze.json'),'variance seal');ck(len(d['rows'])==144 and len({(x['world'],x['model'],x['policy'],x['command'])for x in d['rows']})==144,'144rows')
for path,sha in a['evidence_sha256'].items():ck(h(root/path)==sha,'primary evidence')
for row in d['rows']:
 if row['state']!='complete':continue
 i,m,j,action=row['world'],row['model'],row['command'],row['action'];sl=slice(j*78,(j+1)*78)
 with np.load(b/f'world{i}'/m/'forecasts.npz')as p,np.load(b/f'world{i}/private/held-truth.npz')as t:e=p[action+'_mean'][sl]-t[f'held{j}'].ravel();sd=p[action+'_sd'][sl]
 for kind,var in [('primary',0.),('supplementary',v['method_variance_m2'][m])]:
  w=1.6448536269514722*np.sqrt(sd**2+var);actual={'coverage':np.count_nonzero(abs(e)<=w)/78,'mean_halfwidth_m':w.sum()/78,'mean_interval_score_m':(2*w+20*np.maximum(abs(e)-w,0)).sum()/78}
  for k,x in actual.items():ck(np.isclose(x,row[kind][k],rtol=1e-11,atol=1e-14),kind+k)
 ck(np.isclose(np.linalg.norm(e)/np.sqrt(78),row['position_coordinate_rmse_m'],rtol=1e-11),'same prediction')
for s in d['summaries']:
 rows=[x for x in d['rows']if x['model']==s['model']and x['policy']==s['policy']];valid=[x for x in rows if x['state']=='complete'];ck(s['expected_rows']==12 and s['complete_rows']==len(valid),'summarydenominator')
 for kind in ('primary','supplementary'):
  ck((kind in s)==(len(valid)==12),'no partialmean')
  if len(valid)==12:
   for k in valid[0][kind]:ck(np.isclose(s[kind][k],np.mean([x[kind][k]for x in valid]),rtol=1e-11),'summarymean')
selected=[x for x in a['fit_ledger']if x['action']=='stretch'];ck(len(selected)==12 and all(x['optimizer_success']and not x['incomplete']for x in selected),'all12stretchfits converged')
o={'passed':not errors,'checks':checks,'errors':errors,'created_unix':time.time(),'result_sha256':h(r/'unknown-action-discrepancy-band-results.json'),'primary_audit_sha256':h(r/'unknown-action-final-independent-audit.json'),'fit_counts':a['fit_counts'],'all12_stretch_fits_optimizer_success':all(x['optimizer_success']for x in selected),'selected_policy_summaries':[x for x in d['summaries']if x['policy']=='target_task_variance'],'scope':'Independent all144 savedarray intervalcoverage,width,properintervalscore andsameforecast checks. ScalarvalidationMSE supplement only; not populationcalibration or learnedposterior correctness.'};(r/'discrepancy-band-final-independent-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
