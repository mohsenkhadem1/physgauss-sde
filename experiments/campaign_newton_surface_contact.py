"""Reference-area weighted unilateral plane penalty for stock Newton BE.

This is a declared compliant surface discretization: U=.5*kappa*sum_i
A_i*max(-gap_i,0)^2. A_i are triangle lumped reference areas; kappa has
units N/m^3. It is neither exact nonpenetration nor surface intersection CCD.
No state/velocity/inertial target is modified. Friction and damping are zero.
"""
from __future__ import annotations
import numpy as np

def surface_areas(rest,faces):
 x=np.asarray(rest,dtype=float);f=np.asarray(faces,dtype=int)
 if x.ndim!=2 or x.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite triangular surface required')
 a=.5*np.linalg.norm(np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]]),axis=1)
 if np.any(a<=0):raise ValueError('Degenerate surface')
 weights=np.zeros(len(x));np.add.at(weights,f.ravel(),np.repeat(a/3,3))
 return weights

def penalty(gap,areas,kappa):
 d=np.asarray(gap,float);a=np.asarray(areas,float)
 if d.shape!=a.shape or not np.isfinite(d).all() or not np.isfinite(a).all() or np.any(a<0) or not np.isfinite(kappa) or kappa<=0:raise ValueError('Valid gap/area/stiffness required')
 penetration=np.maximum(-d,0)
 return .5*kappa*np.sum(a*penetration**2),kappa*a*penetration

def area_contact_solver_class(base,wp,areas,kappa):
 """Only replace per-contact material/solve stiffness after native setup.

 Native floor detection/constraint evaluation, prediction, colors, physical
 velocities and objective solve remain stock. The hook rejects barycentric
 E/F contacts or dynamic rigid bodies; scalar contact material must be mu=kd=0.
 """
 areas=np.asarray(areas,float).copy()
 if np.any(areas<0) or not np.isfinite(areas).all() or not np.isfinite(kappa) or kappa<=0:raise ValueError('Invalid area law')
 class AreaContactVBD(base):
  def __init__(self,model,*args,**kwargs):
   if model.body_count!=0 or model.particle_count!=len(areas):raise ValueError('World-plane particle contacts only')
   if kwargs.get('rigid_avbd_beta')!=0. or kwargs.get('rigid_soft_enable_dat') is not False or kwargs.get('particle_enable_self_contact') is not False:raise ValueError('Explicit fixed penalty/no DAT/no selfcontact required')
   if model.soft_contact_mu!=0 or model.soft_contact_kd!=0:raise ValueError('Friction/damping not supported by this declared law')
   super().__init__(model,*args,**kwargs)
  def _refresh_body_particle_contact_state(self,contacts,refresh):
   super()._refresh_body_particle_contact_state(contacts,refresh)
   if contacts is None:return
   n=int(contacts.soft_contact_count.numpy()[0])
   if n==0:return
   ids=contacts.soft_contact_indices.numpy()[:n]
   if np.any(ids[:,1:]!=-1):raise ValueError('Area adapter does not support edge/face contact')
   nodes=ids[:,0]
   if np.any(areas[nodes]<=0):raise ValueError('Interior-node contact: undeclared surface geometry')
   if len(np.unique(nodes))!=len(nodes):raise ValueError('Multiple planes/contact duplicates unsupported')
   kval=(areas[nodes]*kappa).astype(np.float32)
   for field in [self.body_particle_contact_penalty_k,self.body_particle_contact_material_ke]:
    a=field.numpy();a[:n]=kval;field.assign(wp.array(a,dtype=wp.float32,device=self.device))
   self.last_area_contact_ids=nodes.copy();self.last_area_contact_stiffness=kval.copy()
 return AreaContactVBD
