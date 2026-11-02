import json,hashlib,itertools,time
from pathlib import Path
import numpy as np
root=Path.cwd(); r=root/'reports/compositional-neural-mechanics-v16'; out=root/'runs/compositional-neural-mechanics-v16/model-discrepancy-development'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((r/'model-discrepancy-development-plan.json').read_text()); d=json.loads((r/'model-discrepancy-development.json').read_text()); f=json.loads((r/'model-discrepancy-variance-freeze.json').read_text()); errors=[]; checks=0;evidence={}
def check(ok,msg):
 global checks
 checks+=1
 if not ok:errors.append(msg)
def bind(path):evidence[str(path.relative_to(root))]=sha(path)
for name in ['model-discrepancy-development-plan.json','model-discrepancy-development.json','model-discrepancy-variance-freeze.json']:bind(r/name)
check(f['report_sha256']==sha(r/'model-discrepancy-development.json'),'report seal');check(f['plan_sha256']==d['plan_sha256']==sha(r/'model-discrepancy-development-plan.json'),'plan seals')
for table in ['source_sha256','input_sha256']:
 for path,h in p[table].items():check(sha(root/path)==h,'pinned '+path)
expected=set(itertools.product(range(4),p['methods'],range(2)));check(len(d['rows'])==24 and {(x['world'],x['method'],x['action'])for x in d['rows']}==expected,'exact 24 census')
arrays={}
for row in d['rows']:
 key=(row['world'],row['method'],row['action']);path=out/f'{key[0]}-{key[1]}-{key[2]}.npz';bind(path);check(row['completed'] is True and row['response_sha256']==sha(path),'response '+str(key))
 with np.load(path)as a:
  check(all(np.isfinite(a[k]).all()for k in a.files),'finite '+str(key));arrays[key]=a['displacement'].copy();check(arrays[key].shape==(26,3),'shape '+str(key))
results={}
for method in p['methods'][1:]:
 e=np.array([[arrays[i,method,j]-arrays[i,'analytic_reference_n4',j]for j in range(2)]for i in range(4)])
 path=out/f'{method}-signed-errors.npz';bind(path)
 with np.load(path)as a:check(np.array_equal(e,a['errors']),method+' exact residual tensor')
 c=d['calibration'][method];mse=sum(float(np.dot(x.ravel(),x.ravel())/x.size)for x in e.reshape(8,26,3))/8
 check(np.isclose(mse,c['discrepancy_variance_m2'],rtol=1e-14,atol=0),method+' MSE');check(np.isclose(mse,f['method_variance_m2'][method],rtol=1e-14,atol=0),method+' frozen variance')
 check(np.isclose(mse,float(e.var()+e.mean()**2),rtol=1e-14,atol=0),method+' bias decomposition')
 check(c['error_tensor_sha256']==sha(path),method+' error seal');check(np.allclose(np.sqrt(np.mean(e*e,axis=(2,3))),c['case_coordinate_rms_m'],rtol=1e-14,atol=0),method+' case RMS')
 results[method]={'mse_m2':mse,'rms_um':np.sqrt(mse)*1e6,'signed_mean_um':e.mean()*1e6}
check(d['elapsed_seconds']<=p['stage_seconds'],'stage budget');check(d['all24_complete'] is True and f['available'] is True,'availability')
a={'passed':not errors,'checks':checks,'errors':errors,'created_unix':time.time(),'results':results,'evidence_sha256':evidence,'scope':'Independent saved-array recomputation of all24 responses/eight residual fields per method. Supplementary predictive variance only; not empirical calibration, parameter-bias correction, or a change to primary fits/means/policy. No unknown-study targets accessed.'}
(r/'model-discrepancy-final-independent-audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:a[k]for k in ['passed','checks','errors','results']},indent=2))
