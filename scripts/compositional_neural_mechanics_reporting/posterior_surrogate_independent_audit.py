"""Independent response checks and joint importance reconstruction; no FEM solves."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri,logsumexp
from scipy.stats import qmc
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-surrogate'
def main():
 t=time.monotonic();read=lambda p:json.loads(Path(p).read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=read(R/'reference-posterior-surrogate-plan.json');d=read(R/'reference-posterior-surrogate-results.json');errors=[];checks=0;evidence={}
 def ck(v,msg):
  nonlocal checks
  checks+=1
  if not v:errors.append(msg)
 def eq(a,b,msg,atol=1e-9):ck(np.allclose(a,b,rtol=2e-8,atol=atol),msg)
 for table in ('source_sha256','input_sha256'):
  for path,v in p[table].items():ck(h(ROOT/path)==v,'pin '+path)
 for path in [R/'reference-posterior-surrogate-plan.json',R/'reference-posterior-surrogate-results.json',B/'start.json',B/'fixed-polynomials.npz']:evidence[str(path.relative_to(ROOT))]=h(path)
 ck(d['plan_sha256']==h(R/'reference-posterior-surrogate-plan.json')==read(B/'start.json')['plan_sha256'],'plan/start')
 with np.load(B/'fixed-polynomials.npz')as a:center=a['center'];L=a['cholesky'];co={i:a[f'degree{i}']for i in (3,4)}
 def basis(x,deg):return np.array([(x[:,0]/8)**i*(x[:,1]/8)**j for i in range(deg+1)for j in range(deg+1-i)]).T
 obsfile=next(ROOT/k for k in p['input_sha256']if k.endswith('observations-and-development-truth.npz'))
 with np.load(obsfile)as a:y=np.concatenate([a[f'obs_{i}_{k}'].ravel()/s for i in range(2)for k,s in [('forces',.02),('torques',.001)]])
 # Refit each fixed polynomial from the declared existing437 exact responses.
 old=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-extension';od=read(R/'reference-posterior-extension-results.json');xx=[];vv=[]
 with np.load(obsfile)as a:posobs=[a[f'obs_{i}_displacement'].copy()for i in range(2)]
 for key in od['levels']['o25-std1']['node_keys']:
  row=read(old/'nodes'/f'{key}.json');x=np.linalg.solve(L,np.array(row['ratio'])-center)
  if np.linalg.norm(x)>8:continue
  with np.load(old/'nodes'/f'{key}.npz')as a:v=np.r_[np.concatenate([(a[f'{i}_displacement']-posobs[i]).ravel()/50e-6 for i in range(2)]),np.concatenate([a[f'{i}_{k}'].ravel()/s for i in range(2)for k,s in [('forces',.02),('torques',.001)]])]
  xx.append(x);vv.append(v)
 ck(len(xx)==437,'training census')
 for deg in (3,4):eq(np.linalg.lstsq(basis(np.array(xx),deg),vv,rcond=None)[0],co[deg],'coefficient refit')
 rows=d['validation']['rows'];ck(len(rows)==64 and sorted(r['index']for r in rows)==list(range(64)),'64 exact validation')
 def ll(v,s):return -.5*(np.sum(v[:156]**2)+np.sum((np.exp(s)[:,None]*v[156:]-y)**2,axis=1))
 maxima={str(deg):{}for deg in (3,4)}
 for row in rows:
  i=row['index'];path=ROOT/row['path'];ck(row['completed']and row['ratio']==p['validation_ratios'][i]and h(path)==row['sha256'],'exact validation identity/hash');evidence[str(path.relative_to(ROOT))]=h(path)
  with np.load(path)as a:truth=a['response']
  ratio=np.array(row['ratio']);lo=max(-1,np.log(20)-ratio.min());hi=min(1,np.log(10000)-ratio.max())
  def nlp(s):return float(-ll(truth,np.array([s]))[0]+.5*(s*s+np.sum((ratio+s-np.log(500))**2)))
  opt=minimize_scalar(nlp,bounds=(lo,hi),method='bounded',options={'xatol':1e-14});mode=min([lo,hi,opt.x],key=nlp);ss=np.array([s for s in p['validation_s']if lo<=s<=hi]+[mode]);x=np.linalg.solve(L,ratio-center)[None,:]
  for deg in (3,4):
   pred=(basis(x,deg)@co[deg])[0];e=pred-truth;values=dict(position_vector_error=np.linalg.norm(e[:156]),wrench_vector_error=np.linalg.norm(e[156:]),position_component_error=np.max(abs(e[:156])),wrench_component_error=np.max(abs(e[156:])),absolute_loglikelihood_error=np.max(abs(ll(pred,ss)-ll(truth,ss))))
   for k,v in values.items():eq(v,row['checks'][str(deg)][k],'validation '+k,atol=2e-8);ck(v<=p['gates'][k],'validation gate');maxima[str(deg)][k]=max(maxima[str(deg)].get(k,0),float(v))
 ck(d['validation']['passed'],'validation status')
 # Independently regenerate primary full2^20 joint Sobol samples and actual density weights.
 n=2**20;mean=np.array(p['proposal_mean']);P=np.linalg.cholesky(p['proposal_covariance']);g=ndtri(qmc.Sobol(3,scramble=True,seed=p['scramble_seeds'][0]).random_base2(20));x=mean+g@P.T;logw=np.empty(n);rad=np.empty(n);normal=(ndtr(1)-ndtr(-1))*(ndtr(np.log(10000/500))-ndtr(np.log(20/500)))**2
 const=-156*np.log(50e-6*np.sqrt(2*np.pi))-12*np.log(.02*np.sqrt(2*np.pi))-12*np.log(.001*np.sqrt(2*np.pi))
 for start in range(0,n,8192):
  end=min(n,start+8192);z=x[start:end];white=np.linalg.solve(L,(z[:,1:]-z[:,:1]-center).T).T;rad[start:end]=np.linalg.norm(white,axis=1);v=basis(white,4)@co[4];res=v[:,156:]*np.exp(z[:,0,None])-y;loglike=-.5*(np.sum(v[:,:156]**2,1)+np.sum(res**2,1))+const;lp=-.5*np.sum((z-[0,np.log(500),np.log(500)])**2,1)-1.5*np.log(2*np.pi)-np.log(normal);lq=-.5*np.sum(g[start:end]**2,1)-1.5*np.log(2*np.pi)-np.log(np.linalg.det(P));valid=(abs(z[:,0])<=1)&(z[:,1:]>=np.log(20)).all(1)&(z[:,1:]<=np.log(10000)).all(1);logw[start:end]=np.where(valid,lp+loglike-lq,-np.inf)
 Z=logsumexp(logw);w=np.exp(logw-Z);m=w@x;delta=x-m;cov=(delta*w[:,None]).T@delta;interval=[]
 for j in range(3):
  order=np.argsort(x[:,j]);interval.append(np.interp([.05,.95],np.cumsum(w[order]),x[order,j]))
 primary=d['primary_summary']
 for k,v in dict(log_evidence=Z-np.log(n),mean=m,covariance=cov,sd=np.sqrt(np.diag(cov)),interval90=interval,importance_ESS=1/(w@w),tail8_mass=w[rad>8].sum(),tail10_mass=w[rad>10].sum()).items():eq(v,primary[k],'independent primary '+k)
 def cmp(a,b):
  sd=np.array(b['sd']);return dict(mean_change_in_sd=float(np.max(abs(np.array(a['mean'])-b['mean'])/sd)),interval_endpoint_change_in_sd=float(np.max(abs(np.array(a['interval90'])-b['interval90'])/sd[:,None])),covariance_relative_change=float(np.linalg.norm(np.array(a['covariance'])-b['covariance'])/np.linalg.norm(b['covariance'])),log_evidence_change=abs(a['log_evidence']-b['log_evidence']))
 def gate(c):return c['mean_change_in_sd']<.02 and c['interval_endpoint_change_in_sd']<.05 and c['covariance_relative_change']<.02
 expected={f'd{a}-f{b}-s{s}'for a in (3,4)for b in (1,2)for s in p['scramble_seeds']};ck(set(d['integration'])==expected,'16 group census')
 for key,levels in d['integration'].items():
  ck(set(levels)=={'16','18','20'},'three resolutions')
  for power,row in levels.items():
   c=cmp(row,primary);stored=d['checks'][key+'-p'+power]
   for k,v in c.items():eq(v,stored[k],'comparison '+k)
   ck(stored['passed']==bool(gate(c)and c['log_evidence_change']<.002),'integration gate')
  row=levels['20'];ck(d['checks'][key+'-tail']['passed']==bool(row['tail8_mass']<1e-6 and row['tail10_mass']<1e-8),'tail gate')
  for radius,cut in row['tailcut'].items():
   c=cmp(cut,row);stored=d['checks'][key+'-cut'+radius]
   for k,v in c.items():eq(v,stored[k],'tailcut '+k)
   ck(stored['passed']==bool(gate(c)and c['interval_endpoint_change_in_sd']<.005),'tailcut gate')
 c=cmp(primary,od['levels']['o33-std1']['posteriors']['VF']['2049']);stored=d['checks']['original_moments']
 for k,v in c.items():eq(v,stored[k],'old moments '+k)
 ck(stored['passed']==bool(c['mean_change_in_sd']<.02 and c['covariance_relative_change']<.02 and c['log_evidence_change']<.002),'old moments gate');ck(len(d['checks'])==97 and all(v['passed']for v in d['checks'].values())and d['qualified']and d['complete']and not d['failures'],'complete qualification');ck(d['elapsed_seconds']<=p['seconds'],'runtime')
 out=dict(passed=not errors,checks=checks,errors=errors,evidence_sha256=evidence,created_unix=time.time(),audit_seconds=time.monotonic()-t,validation_maxima=maxima,primary_independently_regenerated_samples=n,qualified_developmental_surrogate_reference=d['qualified'],tail_diagnostics={k:{f:v['20'][f]for f in ('tail8_mass','tail10_mass','tail_sample_counts','tail_effective_nodes')}for k,v in d['integration'].items()},scope='Original posterior remains VFunqualified. This separately validated continuous numerical surrogate passes declared finite validation and integration consistency gates. All64 exact residual errors and primary2^20 joint weights/quantiles independently recomputed; all16group reported-gate arithmetic checked, other15 raw QMC integrations not regenerated. The primary has zero sampled points outside radius8; wider proposals provide finite tail samples. Neither empirical tail mass nor shell validation is global tail certification; no empirical coverage,15D,unknownmodel,or exactFEM posterior guarantee.')
 (R/'reference-posterior-surrogate-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','checks','errors','audit_seconds']}))
if __name__=='__main__':main()
