"""Input-only explicit SO(3) carrier and smooth material-coordinate residual.
No changing FEM carrier is used; this is empirical response, not elasticity.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.external_benchmark.data import fps,rigid_map,interpolate_matrix
from scripts.external_benchmark.evaluate import features
SETTINGS=dict(centers=12,sigma_fraction=.25,ridge=1.,cap_multiple=2.,carrier='Prefix-fitted proper SO3 and translation anchored at last valid observed pose',deformation='Prefix-fitted Gaussian material response anchored at last valid observed shape',rotation_unwrap='Backward nearest equivalent rotation-vector branch about last observed pose')

def basis(points,centers,sigma):
 B=np.exp(-np.sum((points[:,None]-centers[None])**2,axis=-1)/(2*sigma*sigma));return B/B.sum(1,keepdims=True)

def unwrap_about_last(rotations):
 reference=rotations[-1];values=[];previous=np.zeros(3)
 for R in rotations[::-1]:
  rv=Rotation.from_matrix((R@reference.T).T).as_rotvec();length=np.linalg.norm(rv)
  axis=rv/length if length>1e-10 else (previous/np.linalg.norm(previous) if np.linalg.norm(previous)>1e-10 else np.array([1.,0,0]))
  choices=[rv+2*np.pi*k*axis for k in range(-2,3)];chosen=min(choices,key=lambda x:np.linalg.norm(x-previous));values.append(chosen);previous=chosen
 return np.asarray(values[::-1])

def predict(data,initial_nodal,locations=None):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 end=int(data['train_end']);nodes=np.asarray(data['nodes']);locations=nodes if locations is None else np.asarray(locations)
 if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Strict prefix required')
 initial=np.asarray(data['observed_initial']);origin=initial.mean(0);centers=nodes[fps(nodes,12)];sigma=.25*float(np.max(np.ptp(nodes,axis=0)))
 B=basis(initial,centers,sigma);query=basis(locations,centers,sigma)
 F=features(data);mu=F[:end].mean(0);sd=F[:end].std(0);sd[sd<1e-5]=1.;Z=np.c_[np.ones(len(F)),(F-mu)/sd]
 validframes=[];rotations=[];centroids=[];localres=[];masks=[]
 for t in range(end):
  valid=np.asarray(data['observed_mask'][t],bool)&np.isfinite(data['observed'][t]).all(1)
  if valid.sum()<4:continue
  R,b=rigid_map(initial[valid],data['observed'][t,valid]);c=origin@R+b
  residual=(data['observed'][t]-c)@R.T-(initial-origin)
  validframes.append(t);rotations.append(R);centroids.append(c);localres.append(residual);masks.append(valid)
 if len(validframes)<4:raise ValueError('Insufficient observed rigid poses')
 vf=np.asarray(validframes);rotations=np.asarray(rotations);centroids=np.asarray(centroids);last=vf[-1];rv=unwrap_about_last(rotations);X=Z[vf];G=X.T@X+np.eye(X.shape[1]);posecoef=np.linalg.solve(G,X.T@np.c_[centroids-centroids[-1],rv]);relative=(Z-Z[last])@posecoef
 Rs=np.asarray([Rotation.from_rotvec(v).as_matrix().T@rotations[-1] for v in relative[:,3:]]);cs=centroids[-1]+relative[:,:3]
 nf=Z.shape[1];normal=np.eye(nf*12);rhs=np.zeros((nf*12,3));cap=1e-6
 for t,y,valid in zip(vf,localres,masks):
  q=B[valid];z=Z[t];normal+=np.kron(np.outer(z,z),q.T@q);rhs+=np.einsum('f,kc->fkc',z,q.T@y[valid]).reshape(nf*12,3);cap=max(cap,2*float(np.linalg.norm(y[valid],axis=1).max()))
 coeff=np.linalg.solve(normal,rhs).reshape(nf,12,3);q=B[masks[-1]];state=np.linalg.solve(q.T@q+np.eye(12),q.T@localres[-1][masks[-1]])
 latent=state[None]+np.einsum('tf,fkc->tkc',Z-Z[last],coeff);deformation=np.einsum('nk,tkc->tnc',query,latent)
 # Use fixed material node probes for amplitude scaling, independent of query resolution.
 probes=np.einsum('nk,tkc->tnc',basis(nodes,centers,sigma),latent);scale=np.minimum(1.,cap/np.maximum(np.linalg.norm(probes,axis=2).max(1),1e-12));deformation*=scale[:,None,None]
 predicted=np.einsum('tni,tij->tnj',locations[None]-origin+deformation,Rs)+cs[:,None]
 # Preserve the exact inherited initial geometry/correspondence convention.
 A=interpolate_matrix(nodes,locations);predicted[0]=locations+A@(np.asarray(initial_nodal)-nodes)
 inv=np.linalg.inv(Z[:end].T@Z[:end]+np.eye(nf));leverage=np.einsum('ti,ij,tj->t',Z,inv,Z)
 return predicted,dict(settings=SETTINGS,fit_end=end,last_pose_frame=int(last),valid_pose_frames=vf.tolist(),sigma_m=sigma,pose_coefficient_norm=float(np.linalg.norm(posecoef)),deformation_coefficient_norm=float(np.linalg.norm(coeff)),deformation_cap_m=cap,amplitude_scale=scale.tolist(),query_count=len(locations),known_command_leverage=leverage.tolist(),scope='Explicit rigid carrier plus empirical smooth material deformation; no FEM equilibrium, force identification, or orientation guarantee')
