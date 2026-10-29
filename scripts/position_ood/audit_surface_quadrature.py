"""Independent higher-order surface integration check for the finite footprint."""
from pathlib import Path
import json,sys
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.smooth_fixture import SmoothDensePositionFEM,footprint
from scripts.position_ood.fixture import controls
from scripts.phystwin.tensor_fixture import basis
from scripts.position_ood.study import OUT
from scripts.phystwin.sequential_reference import write
m=SmoothDensePositionFEM(2,'cpu');g=m.geo;q,w=leggauss(10);replacement=[];changes=[]
for pair,old in zip([('lower','lower'),('lower','upper'),('yminus','yplus')],m.matrices):
 matrices=[]
 for left,kind,oldM in zip([True,False],pair,old):
  M=np.zeros((len(g.rest),len(g.rest)));normalizer=0.
  for iy in range(g.cells[1]):
   for iz in range(g.cells[2]):
    ya,yb=g.axes[1][iy:iy+2];za,zb=g.axes[2][iz:iz+2];yy=(q+1)*(yb-ya)/2+ya;zz=(q+1)*(zb-za)/2+za;v=np.stack(np.meshgrid(yy,zz,indexing='ij'),-1).reshape(-1,2);N,_=basis(np.c_[np.full(len(v),-1. if left else 1.),2*(v[:,0]-ya)/(yb-ya)-1,2*(v[:,1]-za)/(zb-za)-1]);weights=np.outer(w,w).ravel()*(yb-ya)*(zb-za)/4*footprint(v[:,0],v[:,1],kind);e=np.ravel_multi_index((0 if left else g.cells[0]-1,iy,iz),tuple(g.cells));ids=g.elements[e];M[np.ix_(ids,ids)]+=N.T@(weights[:,None]*N);normalizer+=weights.sum()
  M/=normalizer;changes.append(float(np.linalg.norm(M-oldM.numpy())/np.linalg.norm(M)));matrices.append(m.tensor(M))
 replacement.append(matrices)
theta=m.tensor([*np.log([8000.,15000.,15000.]),0,0,-9.81,.15,.5]);poses=[controls(a) for a in ['rest','stretch','bend_z']];before=np.array([m.predict(theta,p,0).numpy() for p in map(m.tensor,poses)]);m.matrices=replacement;after=np.array([m.predict(theta,p,0).numpy() for p in map(m.tensor,poses)])
write(OUT/'surface-quadrature-audit.json',dict(surface_orders=[6,10],mesh=2,scope='Independent higher-order surface assembly; same nonlinear material and controls. Soft material/high coupling development stress check.',relative_matrix_changes=changes,per_action_displacement_rms_change_m=np.sqrt(np.mean((before-after)**2,axis=(1,2))).tolist()))
