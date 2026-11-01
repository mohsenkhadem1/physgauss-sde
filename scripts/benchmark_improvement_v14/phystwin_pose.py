"""Input-only attachment-pose candidates; optional inherited FEM deformation.
No force/material inference: rigid attachment versus rotation-free attachment
are explicit kinematic hypotheses. All calibration uses permitted prefix tracks.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.external_benchmark.data import rigid_map

def observed_pose(data,end):
 for t in range(end-1,-1,-1):
  valid=np.asarray(data['observed_mask'][t],bool)&np.isfinite(data['observed'][t]).all(1)
  if valid.sum()>=3:
   R,b=rigid_map(data['observed_initial'][valid],data['observed'][t,valid]);return t,R,b
 raise ValueError('No valid observed pose')

def predict(data,base_prediction=None,method='controller_translation',rotation_gain=None):
 if 'evaluation' in data:raise ValueError('Evaluator data forbidden')
 end=int(data['train_end']);nodes=np.asarray(data['nodes']);observed=np.asarray(data['observed'])
 if len(observed)!=end or len(data['observed_mask'])!=end:raise ValueError('Strict prefix required')
 controls=np.asarray(data['controls']);T=len(controls)
 if not np.isfinite(controls).all():raise ValueError('Nonfinite known controls')
 if method not in ('controller_translation','controller_rigid','controller_fitted_rotation'):raise ValueError(method)
 k,object_R,object_b=observed_pose(data,end)
 control_pose=[rigid_map(controls[k],frame) for frame in controls]
 if method=='controller_translation':gain=0.
 elif method=='controller_rigid':gain=1.
 else:
  # Scalar relative rotational compliance, fitted on prefix only; bounded
  # between a free-orientation attachment and a fully locked attachment.
  c=[];o=[]
  for t in range(k+1):
   valid=np.asarray(data['observed_mask'][t],bool)&np.isfinite(observed[t]).all(1)
   if valid.sum()<3:continue
   R,_=rigid_map(data['observed_initial'][valid],observed[t,valid]);Q,_=control_pose[t]
   c.append(Rotation.from_matrix(Q.T).as_rotvec());o.append(Rotation.from_matrix((object_R.T@R).T).as_rotvec())
  c=np.asarray(c);o=np.asarray(o);gain=float(np.clip(np.sum(c*o)/max(np.sum(c*c),1e-12),0.,1.))
 if rotation_gain is not None:raise ValueError('External rotation gain override not permitted')
 shape=nodes.copy();com0=nodes.mean(0);last_com=com0@object_R+object_b
 result=[]
 if base_prediction is not None:
  base=np.asarray(base_prediction)
  if base.shape!=(T,)+nodes.shape:raise ValueError('Inherited FEM trajectory mismatch')
 for t,(Q,b) in enumerate(control_pose):
  if base_prediction is not None:
   if not np.isfinite(base[t]).all():result.append(np.full_like(nodes,np.nan));continue
   Rbase,bbase=rigid_map(nodes,base[t]);shape=(base[t]-bbase)@Rbase.T
  # An attachment point follows the known controller centroid. Its offset to
  # object COM rotates only under the declared rotational-lock hypothesis.
  Qgain=Rotation.from_rotvec(Rotation.from_matrix(Q.T).as_rotvec()*gain).as_matrix().T
  center=controls[t].mean(0)+(last_com-controls[k].mean(0))@Qgain
  R=object_R@Qgain
  result.append((shape-com0)@R+center)
 result=np.asarray(result);result[0]=nodes
 return result,dict(method=method,calibration_frame=k,rotation_gain=gain,shape='inherited per-frame FEM deformation' if base_prediction is not None else 'rigid rest shape',scope='Kinematic attachment hypothesis; not a new elastic solve; no learned forces/material labels',prefix_end=end)
