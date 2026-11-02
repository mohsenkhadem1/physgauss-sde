"""Independent saved-node and numerical gate audit; no forward solves."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-extension'
def main():
 h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());p=read(R/'reference-posterior-extension-plan.json');d=read(R/'reference-posterior-extension-results.json');checks=0;errors=[]
 def ck(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:errors.append(msg)
 def eq(x,y,msg):ck(np.allclose(x,y,rtol=2e-9,atol=1e-10),msg)
 ck(d['plan_sha256']==h(R/'reference-posterior-extension-plan.json'),'plan seal');ck(d['proposal_sha256']==h(B/'proposal.json'),'proposal seal')
 for table in ('source_sha256','input_sha256'):
  for path,v in p[table].items():ck(h(ROOT/path)==v,'pin '+path)
 old=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-recovery';proposal=read(B/'proposal.json');prior=read(old/'proposal.json')
 eq(proposal['center'],prior['center'],'same proposal center');eq(proposal['covariance'],prior['covariance'],'same proposal covariance')
 obsfile=next(ROOT/k for k in p['input_sha256']if k.endswith('observations-and-development-truth.npz'))
 with np.load(obsfile)as raw:obs=[{k:raw[f'obs_{i}_{k}'].copy()for k in ('displacement','forces','torques')}for i in range(2)]
 seen=set();reused=0;failed=0
 ck(set(d['levels'])=={'o25-std1','o33-std1','o33-std2'},'fixed completed level set')
 for name,level in d['levels'].items():
  ck(len(level['node_keys'])==level['node_count'] and level['node_count']+level['zero_prior_nodes']==level['order']**2,'level accounting')
  ck(not level['failed_nodes'],'no hidden failed nodes')
  for key in level['node_keys']:
   if key in seen:continue
   seen.add(key);row=read(B/'nodes'/f'{key}.json');ck(row['plan_sha256']==d['plan_sha256'],'node plan');ck(hashlib.sha256(np.array(row['ratio'],dtype='<f8').tobytes()).hexdigest()==key,'ratio keyed node')
   if not row['completed']:failed+=1;continue
   path=B/'nodes'/f'{key}.npz';ck(h(path)==row['response_sha256'],'node array seal')
   if 'reused_from'in row:
    reused+=1;source=ROOT/row['reused_from']['path'];ck(h(source)==row['reused_from']['sha256']and h(path)==h(source.with_suffix('.npz')),'exact cached reuse')
   with np.load(path)as raw:
    ck(all(np.isfinite(raw[k]).all()for k in raw.files),'node finite');pos=np.concatenate([(raw[f'{i}_displacement']-obs[i]['displacement']).ravel()/50e-6 for i in range(2)]);v=np.concatenate([raw[f'{i}_{k}'].ravel()/s for i in range(2)for k,s in [('forces',.02),('torques',.001)]]);y=np.concatenate([obs[i][k].ravel()/s for i in range(2)for k,s in [('forces',.02),('torques',.001)]])
   eq(-.5*np.dot(pos,pos)-pos.size*np.log(50e-6*np.sqrt(2*np.pi)),row['position_loglikelihood'],'position likelihood');eq([np.dot(v,v),np.dot(v,y),np.dot(y,y)],row['wrench_abc'],'wrench likelihood coefficients')
 recomputed={}
 for track in ('V','VF'):
  pairs=[('order25_33','o25-std1','2049','o33-std1','2049'),('proposal_std_doubling','o33-std1','2049','o33-std2','2049')]+[('scale_'+x,x,'1025',x,'2049')for x in ('o25-std1','o33-std1','o33-std2')]
  for label,an,ag,bn,bg in pairs:
   a=d['levels'][an]['posteriors'][track][ag];b=d['levels'][bn]['posteriors'][track][bg];sd=np.array(b['sd']);mean=np.max(abs(np.array(a['mean'])-b['mean'])/sd);end=np.max(abs(np.array(a['interval90'])-b['interval90'])/sd[:,None]);cov=np.linalg.norm(np.array(a['covariance'])-b['covariance'])/np.linalg.norm(b['covariance']);logz=abs(a['log_evidence']-b['log_evidence']);result=dict(log_evidence_change=float(logz),mean_change_in_sd=float(mean),interval_endpoint_change_in_sd=float(end),covariance_relative_change=float(cov),passed=bool(mean<.02 and end<.05 and cov<.02));key=track+'_'+label;stored=d['checks']['initial'][key]
   for k,v in result.items():eq(v,stored[k],key+k)
   recomputed[key]=result
  for name in ('o25-std1','o33-std1','o33-std2'):
   b=d['levels'][name]['posteriors'][track]['2049'];stored=d['checks']['initial'][track+'_adaptive_normalizer_'+name];ck(stored['passed']==(b['adaptive_normalizer_max_relative_difference']<1e-6),'normalizer gate');eq(stored['relative_error'],b['adaptive_normalizer_max_relative_difference'],'normalizer value')
 ck(d['complete']and not d['failures']and not d['qualified'],'completed overall unqualified');ck(reused==1 and len(seen)==2801 and failed==0,'one exact reused node,2801 total,no failed');ck(d['elapsed_seconds']<=p['seconds'],'recovery runtime');ck(d['refinement_estimated_seconds']>p['seconds']-d['elapsed_seconds']and 'refinement_not_run'in d,'bounded refinement omission')
 tracks={t:all(v['passed']for k,v in d['checks']['initial'].items()if k.startswith(t+'_'))for t in ('V','VF')}
 out=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),report_sha256=h(R/'reference-posterior-extension-results.json'),plan_sha256=h(R/'reference-posterior-extension-plan.json'),unique_nodes=len(seen),reused_nodes=reused,failed_nodes=failed,track_numerical_qualification=tracks,overall_numerically_qualified=False,recomputed_checks=recomputed,scope='Independent all-node saved-response likelihood/cache/source checks and all published convergence gates. CDF endpoint quadrature itself not rerun; its adaptive implementation previously audited/tested. Vpasses fixed consistency checks; VFinterval endpoint proposal sensitivity fails. Neither claim establishes coverage or15D posterior validity.')
 (R/'reference-posterior-extension-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('passed','checks','errors','unique_nodes','reused_nodes','track_numerical_qualification')}))
if __name__=='__main__':main()
