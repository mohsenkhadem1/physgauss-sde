"""Two spatially separated Robin patches, independent stiffness, reset probes.

Reuses the validated Q2 Neo-Hookean energy/Newton solver unchanged. New boundary
integration and load construction are explicit and independently tested.
"""
import numpy as np
from scipy import sparse
from scripts.phystwin.tensor_fixture import TensorFixture
LAYOUTS=['parallel_lower','crossed','opposite_y']
ACTIONS=['initial','left_lower_y','right_upper_y','left_upper_z','right_lower_z','twist','stretch','held_oblique']

def points():
 return np.stack(np.meshgrid(np.linspace(0,.24,9),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)

class SequentialFixture(TensorFixture):
 def __init__(self,n,E,Kleft,Kright,layout,device='cuda:0',order=3):
  if n%2:raise ValueError('Even meshes align both half-face patch boundaries')
  super().__init__(n,E,Kleft,'left_full',device=device,order=order,x_power=1.,y_grading=False,z_refinement=1)
  self.layout=layout;self.Kleft=Kleft;self.Kright=Kright
  kinds={'parallel_lower':('lower','lower'),'crossed':('lower','upper'),'opposite_y':('yminus','yplus')}[layout]
  self.fixture_scalar=sparse.csr_matrix((len(self.rest),len(self.rest)))
  for left,K,kind in zip([True,False],[Kleft,Kright],kinds):
   M,_,area=self.patch(left,kind);self.fixture_scalar+=K*M/area
  self.spring_matrix=sparse.kron(self.fixture_scalar,sparse.eye(3),format='csr')
  self.loads={}
  for left in [True,False]:
   for kind in ['full','lower','upper']:
    _,w,area=self.patch(left,kind);self.loads[left,kind]=w/area
  integral=np.einsum('i,j,k->ijk',[1,4,1],[1,4,1],[1,4,1]).ravel()/216
  self.volume_weights=np.zeros(len(self.rest));np.add.at(self.volume_weights,self.elements,self.volumes[:,None]*integral)
  self.volume_weights/=self.volume_weights.sum();self.set_action('initial')
 def patch(self,left,kind):
  cell=np.stack(np.meshgrid(np.arange(self.cells[1]),np.arange(self.cells[2]),indexing='ij'),-1).reshape(-1,2)
  yc=(self.axes[1][cell[:,0]]+self.axes[1][cell[:,0]+1])/2;zc=(self.axes[2][cell[:,1]]+self.axes[2][cell[:,1]+1])/2
  mask={'full':np.ones(len(cell),bool),'lower':zc<0,'upper':zc>0,'yminus':yc<0,'yplus':yc>0}[kind];cell=cell[mask]
  offset=np.stack(np.meshgrid(np.arange(3),np.arange(3),indexing='ij'),-1).reshape(-1,2);yz=2*cell[:,None]+offset
  ix=np.full(yz.shape[:2],0 if left else self.grid_shape[0]-1);nodes=np.ravel_multi_index((ix,yz[:,:,0],yz[:,:,1]),self.grid_shape)
  area=np.diff(self.axes[1])[cell[:,0]]*np.diff(self.axes[2])[cell[:,1]]
  mass=np.array([[4.,2.,-1.],[2.,16.,2.],[-1.,2.,4.]])/15;local=np.kron(mass,mass)/4;integral=np.outer([1.,4.,1.],[1.,4.,1.]).ravel()/36
  rows=np.broadcast_to(nodes[:,:,None],(len(nodes),9,9));cols=np.broadcast_to(nodes[:,None,:],rows.shape)
  M=sparse.coo_matrix(((area[:,None,None]*local).ravel(),(rows.ravel(),cols.ravel())),shape=(len(self.rest),len(self.rest))).tocsr()
  w=np.zeros(len(self.rest));np.add.at(w,nodes,area[:,None]*integral)
  return M,w,float(area.sum())
 def set_action(self,action):
  self.load=np.zeros_like(self.rest);self.load[:,2]=-.02*self.volume_weights
  def add(left,kind,force):self.load[:]+=self.loads[left,kind][:,None]*np.array(force)
  if action=='initial':add(False,'full',[0,.01,0])
  elif action=='left_lower_y':add(True,'lower',[0,.08,0])
  elif action=='right_upper_y':add(False,'upper',[0,.08,0])
  elif action=='left_upper_z':add(True,'upper',[0,0,.08])
  elif action=='right_lower_z':add(False,'lower',[0,0,.08])
  elif action=='twist':add(True,'upper',[0,.06,0]);add(False,'lower',[0,-.06,0])
  elif action=='stretch':add(True,'full',[-.08,0,0]);add(False,'full',[.08,0,0])
  elif action=='held_oblique':add(False,'lower',[.04,-.06,.05]);add(True,'upper',[0,.02,-.03])
  else:raise ValueError(action)
 def sampled(self,state):
  u=state['displacement_local_m'];x=points();v=self.query_field(u,x);g=self.query_field(u,x,True)
  return dict(field=v,strain=(g+g.transpose(0,2,1))/2,residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),elastic_J=state['elastic_J'],spring_J=state['spring_J'],reaction_balance_N=float(np.linalg.norm((self.fixture_scalar@u-self.load).sum(0))))
