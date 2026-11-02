import json,hashlib
from pathlib import Path
import numpy as np
R=Path('reports/compositional-neural-mechanics-v16');B=Path('runs/compositional-neural-mechanics-v16/reference-posterior-extension');report=json.loads((R/'reference-posterior-extension-results.json').read_text());prop=json.loads((B/'proposal.json').read_text());L=np.linalg.cholesky(prop['covariance']);center=np.array(prop['center']);sets={}
with np.load('runs/compositional-neural-mechanics-v16/conditional-inference-pilot/observations-and-development-truth.npz') as a: obs=np.concatenate([a[f'obs_{i}_displacement'].ravel()/50e-6 for i in range(2)])
for name,level in report['levels'].items():
 xs=[];ys=[]
 for key in level['node_keys']:
  row=json.loads((B/'nodes'/f'{key}.json').read_text());p=B/'nodes'/f'{key}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==row['response_sha256']
  with np.load(p) as a:
   y=np.r_[np.concatenate([a[f'{i}_displacement'].ravel()/50e-6 for i in range(2)])-obs,np.concatenate([a[f'{i}_{k}'].ravel()/sigma for i in range(2) for k,sigma in [('forces',.02),('torques',.001)]])]
  xs.append(np.linalg.solve(L,np.array(row['ratio'])-center));ys.append(y)
 sets[name]=(np.array(xs),np.array(ys),level['node_keys'])
x,y,keys=sets['o25-std1'];train=np.linalg.norm(x,axis=1)<=8;out={'scope':'Cache-only forward-response smoothness diagnostic, not posterior convergence or surrogate certification','training':'o25-std1, whitened Euclidean radius <=8','degrees':{},'tail_note':'Outside radius8 reported separately, no posterior tail bound asserted.'}
for degree in (2,3,4):
 powers=[(i,j)for i in range(degree+1)for j in range(degree+1-i)]
 def design(xx):return np.stack([(xx[:,0]/8)**i*(xx[:,1]/8)**j for i,j in powers],axis=1)
 coef=np.linalg.lstsq(design(x[train]),y[train],rcond=None)[0];rows={}
 for name,(xx,yy,kk)in sets.items():
  if name=='o25-std1':continue
  err=design(xx)@coef-yy;rad=np.linalg.norm(xx,axis=1);ind=np.array([k not in set(keys) for k in kk])
  for label,mask in [('r<=6',rad<=6),('6<r<=8',(rad>6)&(rad<=8)),('r>8',rad>8)]:
   mask=mask&ind;e=err[mask];rows[name+':'+label]={'n':int(mask.sum()),'position_residual_L2_error_max':float(np.max(np.linalg.norm(e[:,:156],axis=1))),'position_component_error_max':float(np.abs(e[:,:156]).max()),'wrench_normalized_L2_error_max':float(np.max(np.linalg.norm(e[:,156:],axis=1))),'wrench_component_error_max':float(np.abs(e[:,156:]).max())}
 out['degrees'][str(degree)]={'training_nodes':int(train.sum()),'basis_terms':len(powers),'design_condition_number':float(np.linalg.cond(design(x[train]))),'validation':rows}
print(json.dumps(out,indent=2))
