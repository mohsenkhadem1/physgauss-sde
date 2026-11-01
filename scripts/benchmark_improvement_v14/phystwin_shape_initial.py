"""Prefix-calibrated scalar FEM deformation response, with inherited volume guard."""
import numpy as np
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.object_frame import adapt,_observed_prediction
from scripts.contact_benchmark_v12.feasibility import repair,inherited_elements

def predict(data,base_prediction=None,method='shape_gain_pose'):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 end=int(data['train_end']);nodes=np.asarray(data['nodes']);base=np.asarray(base_prediction)
 if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Strict prefix required')
 if base.shape!=(len(data['times']),)+nodes.shape:raise ValueError('Exact inherited trajectory required')
 if method not in ('shape_gain_pose','shape_gain_residual','shape_gain_none'):raise ValueError(method)
 posemethod=method.removeprefix('shape_gain_');posed,posemeta=adapt(data,base,method=posemethod,strength=1.)
 rigid=[];deformation=[]
 for t,points in enumerate(posed):
  if not np.isfinite(points).all():rigid.append(points.copy());deformation.append(points.copy());continue
  R,b=rigid_map(nodes,points);rigid.append(nodes@R+b);deformation.append(points-(nodes@R+b))
 rigid=np.asarray(rigid);deformation=np.asarray(deformation)
 rp=_observed_prediction(data,rigid[:end]);dp=np.einsum('pn,tnc->tpc',data['observation_matrix'],deformation[:end])
 valid=np.asarray(data['observed_mask'],bool)&np.isfinite(data['observed']).all(2)&np.isfinite(rp).all(2)&np.isfinite(dp).all(2)
 if not valid.any():raise ValueError('No valid deformation calibration')
 x=dp[valid];y=(data['observed']-rp)[valid]
 # A single response multiplier has direct modal interpretation; bounded to
 # interpolation/modest extrapolation, with explicit local-volume repair.
 gain=float(np.clip(np.sum(x*y)/max(np.sum(x*x),1e-12),0.,2.))
 proposal=rigid+gain*deformation;proposal[0]=nodes
 result,guard=repair(data,base,proposal,elements=inherited_elements(data))
 return result,dict(method=method,deformation_gain=gain,prefix_end=end,pose=posemeta,guard=guard,scope='One empirical response gain on inherited FEM deformation; not material identification, fullcoupled contact, or a new physicalsolve')
