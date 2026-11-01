"""One fixed smooth spatial residual hypothesis; input-only, not a FEM solve."""
import numpy as np
from scripts.external_benchmark.data import fps
from scripts.contact_benchmark_v12.object_frame import adapt,control_frames,_observed_prediction,project_deformation

SETTINGS=dict(centers=12,sigma_fraction=.25,ridge=1.,strength=.5,cap_multiple=2.)

def predict(data,base_prediction):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 end=int(data['train_end']);nodes=np.asarray(data['nodes']);base=np.asarray(base_prediction)
 if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Strict observed prefix required')
 if len(nodes)<12:raise ValueError('Twelve rest centers required')
 out,pose=adapt(data,base,method='pose',strength=.5);frames=control_frames(data);active=np.asarray(data['attachment_active']).sum(1)>0
 centers=nodes[fps(nodes,12)];sigma=.25*float(np.max(np.ptp(nodes,axis=0)))
 B=np.exp(-np.sum((nodes[:,None]-centers[None])**2,axis=2)/(2*sigma*sigma));B/=B.sum(1,keepdims=True)
 O=np.asarray(data['observation_matrix'])@B
 features=[]
 for t,(R,b) in enumerate(frames[:len(base)]):
  local=(data['controller_targets'][t]-b)@R.T-nodes[None];features.append(np.r_[local.mean(1).ravel(),np.array([0.,0.,-1.])@R.T])
 F=np.asarray(features);mu=F[:end].mean(0);sd=F[:end].std(0);sd[sd<1e-5]=1.;Z=np.c_[np.ones(len(F)),(F-mu)/sd]
 observed=_observed_prediction(data,out[:end]);nf=Z.shape[1];G=np.eye(nf*12);rhs=np.zeros((nf*12,3));cap=1e-6;count=0
 for t,(R,b) in enumerate(frames[:end]):
  residual=(data['observed'][t]-observed[t])@R.T
  valid=np.asarray(data['observed_mask'][t],bool)&np.isfinite(residual).all(1)&active[t]
  if not valid.any():continue
  q=O[valid];y=residual[valid];z=Z[t]
  G+=np.kron(np.outer(z,z),q.T@q);rhs+=np.einsum('f,kc->fkc',z,q.T@y).reshape(nf*12,3)
  cap=max(cap,2*float(np.linalg.norm(y,axis=1).max()));count+=int(valid.sum())
 coef=np.linalg.solve(G,rhs).reshape(nf,12,3);nodal=np.einsum('tf,fkc,nk->tnc',Z,coef,B)
 scales=np.minimum(1.,cap/np.maximum(np.linalg.norm(nodal,axis=2).max(1),1e-12));nodal*=scales[:,None,None]
 for t,(R,b) in enumerate(frames[:len(base)]):
  if active[t] and t>0:
   local=(out[t]-b)@R.T;correction=project_deformation(local,nodal[t]*.5);out[t]+=correction@R
 out[0]=base[0]
 return out,dict(settings=SETTINGS,prefix_end=end,centers=centers.tolist(),sigma_m=sigma,valid_marker_frames=count,cap_m=cap,amplitude_scales=scales.tolist(),coefficient_norm=float(np.linalg.norm(coef)),pose=pose,scope='Smooth empirical controller-conditioned Gaussian residual, not physical equilibrium or calibrated materials')
