"""Two independently moving deformable volumes with reciprocal surface contact."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM
from scripts.complex_scenes_v18.pair_contact import inside_closed_surface,closest_squared

class PairContact:
 def __init__(self,model,penalty=4e7):
  elements=model.elements.cpu().numpy();rest=model.rest.cpu().numpy();faces={}
  for tet in elements:
   for opposite in range(4):
    tri=[tet[j] for j in range(4) if j!=opposite];a,b,c=rest[tri]
    if np.cross(b-a,c-a)@(rest[tet[opposite]]-a)>0:tri.reverse()
    key=tuple(sorted(tri));faces[key]=None if key in faces else tri
  tri=np.array([v for v in faces.values() if v is not None]);ids=np.unique(tri);lookup=np.full(len(rest),-1);lookup[ids]=np.arange(len(ids));self.ids=torch.tensor(ids,device=model.device);self.tri=torch.tensor(lookup[tri],device=model.device)
  area=.5*np.linalg.norm(np.cross(rest[tri[:,1]]-rest[tri[:,0]],rest[tri[:,2]]-rest[tri[:,0]]),axis=-1);w=np.zeros(len(ids));np.add.at(w,lookup[tri].ravel(),np.repeat(area/3,3));self.weights=model.t(w);self.penalty=penalty
 def evaluate(self,nodes):
  surface=[x[self.ids] for x in nodes]
  if bool(((surface[0].max(0).values<surface[1].min(0).values)|(surface[1].max(0).values<surface[0].min(0).values)).any()):return [torch.zeros_like(x) for x in nodes],0.,0.
  x=[v.detach().requires_grad_(True) for v in surface];energies=[];depth=0.
  for i,j in [(0,1),(1,0)]:
   inside=inside_closed_surface(x[i],x[j],self.tri)
   if bool(inside.any()):
    dsq=closest_squared(x[i][inside],x[j],self.tri);energies.append(.25*self.penalty*(self.weights[inside]*dsq).sum());depth=max(depth,float(dsq.detach().max().sqrt()))
  if not energies:return [torch.zeros_like(v) for v in nodes],0.,0.
  energy=sum(energies);grad=torch.autograd.grad(energy,x);forces=[torch.zeros_like(v).index_add(0,self.ids,-g.detach()) for v,g in zip(nodes,grad)]
  return forces,float(energy.detach()),depth

def rollout(m,parameters,kind,dt=.0001,duration=.36):
 p=m.t(parameters).clone();p[1:10]=0.;contact=PairContact(m);x=[m.rest.clone(),m.rest+m.t([.012,.004,.10])];v=[torch.zeros_like(q) for q in x];states=[];times=[];minJ=1.;maxdepth=0.;maxnet=0.
 for step in range(round(duration/dt)+1):
  force=[]
  for xx,vv in zip(x,v):
   f,_,J=m.force(xx,vv,step*dt,p,'calibration',kind);force.append(f);minJ=min(minJ,float(J.min()))
  pair,_,depth=contact.evaluate(x);maxdepth=max(maxdepth,depth);maxnet=max(maxnet,float(sum((f.sum(0) for f in pair)).norm()))
  if step%round(.002/dt)==0:states.append(torch.stack(x));times.append(step*dt)
  if step==round(duration/dt):break
  v=[(vv+dt*(f+ff)/m.mass[:,None]).detach() for vv,f,ff in zip(v,force,pair)];x=[(xx+dt*vv).detach() for xx,vv in zip(x,v)]
 return dict(x=torch.stack(states).cpu().numpy(),times=np.asarray(times),minimum_J=minJ,max_pair_penetration_m=maxdepth,max_net_pair_force_N=maxnet)

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='2');p.add_argument('--fit');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2);m=DynamicFEM(device='cuda:'+a.gpu);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.fit:
  fit=json.loads(Path(a.fit).read_text());parameters=fit['parameters'];kind=fit['kind'];seal=hashlib.sha256(Path(a.fit).read_bytes()).hexdigest()
 else:
  from scripts.solver_driven_gaussians.reference import reference_parameters
  parameters=reference_parameters();kind='reference';seal=None
 start=time.time();r=rollout(m,parameters,kind);np.savez_compressed(out/'pair_drop.npz',x=r.pop('x'),times=r.pop('times'));r.update(completed=True,seconds=time.time()-start,fit_sha256=seal,scope='Two identical-material object instances; separate states, inferred material reused, released attachments, reciprocal sampled finite-penalty contact. No friction or CCD.');(out/'pair_drop.json').write_text(json.dumps(r,indent=2));print(r)
if __name__=='__main__':main()
