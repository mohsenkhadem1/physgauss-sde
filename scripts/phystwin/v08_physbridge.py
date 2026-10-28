"""Stronger-load PhysTwin bridge with explicitly matched policy prefix observations."""
from pathlib import Path
import numpy as np
from scripts.phystwin.sequential_fixture import points,ACTIONS
BASE=Path(__file__).resolve().parents[2]/'runs/phystwin-informative-v08'
from scripts.phystwin.v06_camera import render,track

def masses():
 w=np.ones((9,3,3));w[[0,-1],:,:]*=.5;w[:,[0,-1],:]*=.5;w[:,:,[0,-1]]*=.5;return .96*w.ravel()/w.sum()
def forces(action):
 x=points();f=np.zeros_like(x);f[:,2]=-.02*masses()/.96
 def add(left,kind,force):
  w=np.zeros((9,3,3));yz=np.outer([1.,2.,1.],{'full':[1.,2.,1.],'lower':[1.,1.,0.],'upper':[0.,1.,1.]}[kind]);w[0 if left else -1]=yz;w=w.ravel()/w.sum();f[:]+=w[:,None]*np.array(force)
 if action=='initial':add(False,'full',[0,.01,0])
 elif action=='left_lower_y':add(True,'lower',[0,.08,0])
 elif action=='right_upper_y':add(False,'upper',[0,.08,0])
 elif action=='left_upper_z':add(True,'upper',[0,0,.08])
 elif action=='right_lower_z':add(False,'lower',[0,0,.08])
 elif action=='twist':add(True,'upper',[0,.06,0]);add(False,'lower',[0,-.06,0])
 elif action=='stretch':add(True,'full',[-.08,0,0]);add(False,'full',[.08,0,0])
 elif action=='held_oblique':add(False,'lower',[.04,-.06,.05]);add(True,'upper',[0,.02,-.03])
 else:raise ValueError(action)
 gravity=np.zeros_like(f);gravity[:,2]=-.02*masses()/.96
 return gravity+(16 if action=='initial' else 4)*(f-gravity)

def dataset(obj,actions,observe=True,split='test'):
 rest=points();ref=np.load(BASE/'fields'/f'{split}-{obj}-reference.npz')['field'];visible=np.isclose(rest[:,1],-.04);ids=np.flatnonzero(visible)
 end=np.isclose(rest[:,0],0)|np.isclose(rest[:,0],.24);controls=rest[end].copy();controls[:,0]+=np.where(controls[:,0]<.12,-.02,.02)
 # Slow ramp/hold/reset cycles; interpolated equilibrium targets, not dynamic truth.
 t=np.arange(31)/5;alpha=np.minimum(t/1.4,1.);alpha=np.where(t>3.4,np.maximum(0,1-(t-3.4)/1.4),alpha)
 truth=[rest];loads=[np.zeros_like(rest)];fitted=[rest];rng=np.random.default_rng((1200 if split=='development' else 2200)+obj);hold=[]
 for action in actions:
  for k,amp in enumerate(alpha[1:],1):
   pts=rest+amp*ref[ACTIONS.index(action)];truth.append(pts);loads.append(amp*forces(action));row=rest.copy()
   if observe:
    rgb,depth,cam=render(pts,0,ids);row[ids]=track(rgb,depth,cam,ids,rng)
   fitted.append(row)
   if 2.4<=t[k]<=3.4:hold.append(len(truth)-1)
 n=len(truth)
 data=dict(object_points=np.array(fitted,dtype='float32'),object_colors=np.ones((n,len(rest),3),dtype='float32')*.5,object_visibilities=np.broadcast_to(visible,(n,len(rest))).copy(),object_motions_valid=np.broadcast_to(visible,(n-1,len(rest))).copy(),controller_points=np.broadcast_to(controls,(n,len(controls),3)).astype('float32').copy(),surface_points=np.empty((0,3),dtype='float32'),interior_points=np.empty((0,3),dtype='float32'))
 return data,np.array(loads),np.array(truth),np.array(hold)

def restrict_selection_prefix(data,measurement):
 """Expose only the shared nine prefix coordinates to selection ensembles."""
 rest=points();ids=np.array([np.argmin(np.linalg.norm(rest-[.12,-.04,z],axis=1)) for z in [-.04,0,.04]])
 y=np.asarray(measurement).reshape(3,3);assert data['object_points'].shape==(31,81,3)
 t=np.arange(31)/5;alpha=np.minimum(t/1.4,1.);alpha=np.where(t>3.4,np.maximum(0,1-(t-3.4)/1.4),alpha)
 data['object_points'][:]=rest;data['object_points'][:,ids]=rest[ids]+alpha[:,None,None]*y
 data['object_visibilities'][:]=False;data['object_visibilities'][:,ids]=True
 data['object_motions_valid'][:]=False;data['object_motions_valid'][:,ids]=True
 return data
