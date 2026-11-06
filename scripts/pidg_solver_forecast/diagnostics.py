"""Sampled geometry diagnostics, independent of image error or target states."""
import numpy as np
import torch
from scripts.complex_scenes_v18.pair_contact import inside_closed_surface,closest_squared
class Interpenetration:
 def __init__(self,elements,labels,device='cuda'):
  self.surfaces=[]
  for label in np.unique(labels):
   e=elements[labels==label];tri=np.concatenate([e[:,[1,2,3]],e[:,[0,3,2]],e[:,[0,1,3]],e[:,[0,2,1]]]);keys=np.sort(tri,axis=1);_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True);tri=tri[first[count==1]];ids=np.unique(tri);ids=ids[np.linspace(0,len(ids)-1,min(256,len(ids))).astype(int)];self.surfaces.append((torch.tensor(ids,device=device),torch.tensor(tri,device=device)))
 def __call__(self,x):
  count=0;total=0;depth=0.
  for i,(ids,_) in enumerate(self.surfaces):
   for j,(_,faces) in enumerate(self.surfaces):
    if i==j:continue
    vertices=x[torch.unique(faces)];points=x[ids];candidate=((points>=vertices.min(0).values)&(points<=vertices.max(0).values)).all(-1);total+=len(points)
    if not bool(candidate.any()):continue
    inside=inside_closed_surface(points[candidate],x,faces);count+=int(inside.sum())
    if bool(inside.any()):depth=max(depth,float(closest_squared(points[candidate][inside],x,faces).max().sqrt()))
  return dict(sampled_interbody_inside_fraction=count/max(total,1),sampled_interbody_max_depth_normalized=depth,segmented_components=len(self.surfaces))
