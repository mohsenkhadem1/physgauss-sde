"""Localized inherited-tetrahedron repair with the unchanged global safeguard.
This is saved-frame geometric feasibility, not elastic equilibrium or CCD.
"""
import numpy as np
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.feasibility import repair,jacobians,inherited_elements
from scripts.contact_benchmark_v12.object_frame import adapt,_observed_prediction

CUTOFF=.03+1e-8

def local_repair(data,base,proposal,elements=None):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 base=np.asarray(base);proposal=np.asarray(proposal);nodes=np.asarray(data['nodes'])
 if base.shape!=proposal.shape or base.ndim!=3 or base.shape[1:]!=nodes.shape:raise ValueError('Mismatched nodal trajectories')
 elements=inherited_elements(data) if elements is None else np.asarray(elements)
 result=proposal.copy();rows=[]
 for t,(anchor,desired) in enumerate(zip(base,proposal)):
  row=dict(frame=t)
  if not np.isfinite(anchor).all() or not np.isfinite(desired).all() or not np.all(jacobians(nodes,anchor,elements)>CUTOFF):
   result[t]=np.nan;row.update(failed=True,reason='Invalid inherited anchor or proposal');rows.append(row);continue
  originalJ=jacobians(nodes,desired,elements)
  if np.all(originalJ>CUTOFF):row.update(failed=False,modified_nodes=0,iterations=0,min_J=float(originalJ.min()),global_fallback=False);rows.append(row);continue
  R,b=rigid_map(anchor,desired);aligned=anchor@R+b;delta=desired-aligned;weights=np.ones(len(nodes));trial=desired.copy()
  # Per-vertex weights only decrease. All inherited elements are rechecked,
  # since local attenuation can alter neighboring tetrahedra.
  for iteration in range(64):
   J=jacobians(nodes,trial,elements);bad=~np.isfinite(J)|(J<=CUTOFF)
   if not bad.any():break
   affected=np.unique(elements[bad]);weights[affected]*=.5;weights[weights<2.**-20]=0.
   trial=aligned+weights[:,None]*delta
  J=jacobians(nodes,trial,elements);fallback=not np.all(np.isfinite(J)&(J>CUTOFF))
  if fallback:
   one=dict(data,times=np.asarray(data['times'])[t:t+1]);fixed,metadata=repair(one,anchor[None],trial[None],elements=elements);trial=fixed[0];J=jacobians(nodes,trial,elements) if np.isfinite(trial).all() else np.array([np.nan])
  failed=not np.all(np.isfinite(J)&(J>CUTOFF));result[t]=np.nan if failed else trial
  row.update(failed=failed,modified_nodes=int(np.count_nonzero(weights<1)),iterations=iteration+1,min_J=float(np.nanmin(J)) if np.isfinite(J).any() else None,global_fallback=fallback,min_proposal_J=float(np.nanmin(originalJ)) if np.isfinite(originalJ).any() else None,correction_rms_m=float(np.sqrt(np.mean((trial-desired)**2))) if not failed else None)
  rows.append(row)
 return result,dict(method='Localized residual attenuation plus unchanged global safeguard',cutoff=CUTOFF,max_iterations=64,rows=rows,failed_frames=[r['frame'] for r in rows if r['failed']],limitations=['Same inherited topology; no remeshing','Saved-frame local orientation only, not global self-contact or equilibrium','Spatially varying repair may alter velocities and rigid pose'])

def predict(data,base_prediction=None,method='local_residual',strength=.5):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 if strength not in (.25,.5,1.):raise ValueError('Fixed strength shortlist only')
 if method not in ('local_residual','local_response_residual'):raise ValueError('Unknown localized candidate')
 end=int(data['train_end'])
 if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Strict prefix required')
 base=np.asarray(base_prediction);proposal,adapter=adapt(data,base,method='residual',strength=strength)
 gain=1.
 if method=='local_response_residual':
  posed,_=adapt(data,base,method='pose',strength=strength);delta=proposal-posed
  pp=_observed_prediction(data,posed[:end]);dp=np.einsum('pn,tnc->tpc',data['observation_matrix'],delta[:end]);valid=np.asarray(data['observed_mask'],bool)&np.isfinite(dp).all(2)&np.isfinite(pp).all(2)&np.isfinite(data['observed']).all(2)
  x=dp[valid];y=(data['observed']-pp)[valid]
  gain=float(np.clip(np.sum(x*y)/max(np.sum(x*x),1e-12),0.,2.));proposal=posed+gain*delta;proposal[0]=base[0]
 result,guard=local_repair(data,base,proposal)
 return result,dict(method=method,strength=strength,response_gain=gain,adapter=adapter,guard=guard,prefix_end=end,scope='Empirical residual with localized saved-frame volume repair; not elastic inference or coupled contact solver')
