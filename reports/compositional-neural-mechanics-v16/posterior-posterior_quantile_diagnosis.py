import json,math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scripts.compositional_neural_mechanics.reference_posterior_recovery_math import ConditionalCDF
R=Path('reports/compositional-neural-mechanics-v16');B=Path('runs/compositional-neural-mechanics-v16/reference-posterior-extension');res=json.loads((R/'reference-posterior-extension-results.json').read_text());prop=json.loads((B/'proposal.json').read_text());out={}
for factor in (1,2):
 key=f'o33-std{factor}';level=res['levels'][key];rows=[json.loads((B/'nodes'/f'{k}.json').read_text()) for k in level['node_keys']];q,w=hermgauss(33);aa,bb=np.meshgrid(q,q,indexing='ij');v=np.stack([aa.ravel(),bb.ravel()],1);L=np.linalg.cholesky(np.array(prop['covariance'])*factor**2);r=np.array(prop['center'])+np.sqrt(2)*v@L.T;lw=np.log(w[:,None]*w[None,:]/np.pi).ravel()+np.log(2*np.pi)+np.log(np.linalg.det(L))+np.sum(v*v,axis=1)
 cs=[ConditionalCDF(rr,row['wrench_abc'],'VF') for rr,row in zip(r,rows)];lm=np.array([ww+row['position_loglikelihood']+c.logZ-.5*row['wrench_abc'][2]+.5*row['wrench_abc'][1]**2/row['wrench_abc'][0] for ww,row,c in zip(lw,rows,cs)]);mass=np.exp(lm-logsumexp(lm));idx=np.argsort(-mass);keep=idx[np.cumsum(mass[idx])<=.999999];keep=np.r_[keep,idx[len(keep)]]
 widths=np.array([1/math.sqrt(3+2*c.A*math.exp(2*c.mode)-c.B*math.exp(c.mode))for c in cs]);cross={}
 for f in (1,2):
  ends=res['levels'][f'o33-std{f}']['posteriors']['VF']['2049']['interval90'][1]
  cross[str(f)]=[float(sum(mass[k]*cs[k](t-r[k,0])for k in keep)) for t in ends]
 out[key]=dict(conditional_local_sd_mass_average=float(mass@widths),central_left_ratio_spacing=float(np.sqrt(2)*L[0,0]*(q[17]-q[16])),effective_nodes=float(1/(mass@mass)),dominant_nodes_999999=len(keep),leftK_CDF_at_each_report_endpoints=cross,interval90=level['posteriors']['VF']['2049']['interval90'],sd=level['posteriors']['VF']['2049']['sd'],normalizer_relative_error=level['posteriors']['VF']['2049']['adaptive_normalizer_max_relative_difference'])
print(json.dumps(out,indent=2))
