import json,pathlib,numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.integrate import cumulative_simpson
from scipy.linalg import cholesky
from scripts.compositional_neural_mechanics.reference_posterior_math import scale_bounds,LOGK_MEAN
base=pathlib.Path('runs/compositional-neural-mechanics-v16/reference-posterior');p=json.load(open(base/'proposal.json'));q,w=hermgauss(9);v=np.stack(np.meshgrid(q,q,indexing='ij'),-1).reshape(-1,2);L=cholesky(p['covariance'],lower=True);nodes=p['center']+np.sqrt(2)*v@L.T
import hashlib
rows=[json.load(open(base/'nodes'/f"{hashlib.sha256(np.asarray(r,dtype='<f8').tobytes()).hexdigest()}.json")) for r in nodes]
logw=np.log((w[:,None]*w[None,:]/np.pi).ravel())+np.log(2*np.pi)+np.log(np.linalg.det(L))+np.sum(v*v,1)
for points in (1025,2049,4097):
 s=np.linspace(-1,1,points)[None,:]+np.zeros((81,1));x=s[:,:,None]+np.stack([np.zeros(81),nodes[:,0],nodes[:,1]],1)[:,None,:]
 prior=-.5*np.sum((x-[0,LOGK_MEAN,LOGK_MEAN])**2,axis=-1)
 for track in ('V','VF'):
  lp=prior+np.array([r['position_loglikelihood'] for r in rows])[:,None]
  A,B,C=np.array([r['wrench_abc']for r in rows]).T
  if track=='VF':lp-=.5*(A[:,None]*np.exp(2*s)-2*B[:,None]*np.exp(s)+C[:,None])
  coef=np.ones(points);coef[1:-1:2]=4;coef[2:-1:2]=2
  Z=logsumexp(logw[:,None]+lp+np.log(coef*2/(points-1)/3));pdf=np.exp(logw[:,None]+lp-Z)
  cdf=cumulative_simpson(pdf,x=s,axis=1,initial=0);delta=np.diff(cdf,axis=1)
  print(track,points,'mindCDF',delta.min(),'negative_mass',-delta[delta<0].sum(),'width_s',np.min(1/np.sqrt(A)*(A/B)),np.max(1/np.sqrt(A)*(A/B)))
from scripts.compositional_neural_mechanics.reference_posterior_recovery_math import integrate as repaired
from scripts.compositional_neural_mechanics.reference_posterior_math import integrate as original,compare
records={}
for track in ('V','VF'):
 args=(nodes,logw,[r['position_loglikelihood']for r in rows],[r['wrench_abc']for r in rows],0.,track)
 lo=repaired(*args,1025);hi=repaired(*args,2049);oldfine=original(*args,4097)
 records[track]=dict(recovered_1025=lo,recovered_2049=hi,scale_check=compare(lo,hi),versus_original_4097=compare(hi,oldfine))
report=dict(scope='Cached initial81-node integration diagnostic only; no new forward solves',records=records,source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [pathlib.Path(__file__),pathlib.Path('scripts/compositional_neural_mechanics/reference_posterior_recovery_math.py'),base/'proposal.json']},
 failure={'V_negative_CDF_increment_1025':0.,'VF_negative_CDF_increment_1025':-2.8676241093927857e-6,'VF_total_negative_mass_1025':1.8002755081318883e-5,'VF_negative_CDF_increment_2049':-3.93574062229618e-14},
 explanation='Partial cumulative Simpson uses signed quadratic interpolation weights; with a dimensionless scale width of~0.00305 and grid spacing~0.00195 this creates a genuinely nonmonotone CDF. Full-interval Simpson has positiveweights. Repaired conditionalCDF directly integrates positive original density; no clamping or convergence-gate relaxation.')
pathlib.Path('reports/compositional-neural-mechanics-v16/reference-posterior-cdf-diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print({t:records[t]['scale_check']for t in records})
