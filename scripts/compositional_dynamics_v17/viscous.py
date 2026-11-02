"""Objective internal Kelvin–Voigt stress and explicit-midpoint dynamics.

Fixed synthetic eta/zeta are Pa*s, not fitted restitution or global drag.
Cauchy stress =2 eta dev(D)+zeta tr(D) I, D=sym(grad_X(v) F^-1).
Reference Piola J sigma F^-T; power/reference-volume J sigma:D >=0.
No old operator or conservative trajectory is changed.
"""
import numpy as np
from scripts.compositional_dynamics_v17.dynamics import VelocityVerlet,State

class ViscousMidpoint(VelocityVerlet):
 def __init__(self,model,bodies=(),eta=20.,zeta=40.):
  super().__init__(model,bodies)
  if not np.isfinite([eta,zeta]).all()or min(eta,zeta)<0:raise ValueError('Finite nonnegative viscosities required')
  self.eta=float(eta);self.zeta=float(zeta);self.grad=model.engine.grad.detach().cpu().numpy();self.weights=model.engine.weights.detach().cpu().numpy();self.dissipated_J=0.
 def viscous(self,u,v):
  ids=self.model.elements;F=np.eye(3)+np.einsum('eai,qeaj->qeij',u[ids],self.grad);J=np.linalg.det(F)
  if not np.isfinite(F).all()or not np.isfinite(v).all()or J.min()<=self.model.minimum_J:raise ValueError('Finite positive-J viscous state required')
  inv=np.linalg.inv(F);V=np.einsum('eai,qeaj->qeij',v[ids],self.grad);L=V@inv;D=.5*(L+L.swapaxes(-1,-2));tr=np.trace(D,axis1=-2,axis2=-1);dev=D-tr[...,None,None]*np.eye(3)/3
  sigma=2*self.eta*dev+self.zeta*tr[...,None,None]*np.eye(3);P=J[...,None,None]*(sigma@inv.swapaxes(-1,-2));local=-np.einsum('qe,qeiJ,qeaJ->eai',self.weights,P,self.grad);force=np.zeros_like(u);np.add.at(force,ids,local)
  power=float(np.sum(self.weights*J*(2*self.eta*np.sum(dev*dev,axis=(-2,-1))+self.zeta*tr*tr)))
  if not np.isfinite(force).all()or not np.isfinite(power)or power<0:raise ValueError('Invalid viscous force/dissipation')
  return force,power
 def evaluate_viscous(self,u,v,spheres):
  force,bf,bt,potential,diag=self.evaluate(u,spheres);fv,power=self.viscous(u,v);diag=dict(diag,viscous_power_W=power);return force+fv,bf,bt,potential,diag
 def initial_state(self,u=None,v=None):
  s=super().initial_state(u,v);s.force,s.body_forces,s.body_torques,s.potential,s.diagnostics=self.evaluate_viscous(s.u,s.v,s.spheres);self.dissipated_J=0.;return s
 def step(self,s,dt):
  if not np.isfinite(dt)or dt<=0:raise ValueError('Finite positive timestep')
  um=s.u+.5*dt*s.v;vm=s.v+.5*dt*self.acceleration(s.force);bm={}
  for n,b in s.spheres.items():bm[n]=dict(center=b['center']+.5*dt*b['velocity'],velocity=b['velocity']+.5*dt*s.body_forces[n]/self.bodies[n].mass,omega=b['omega']+.5*dt*s.body_torques[n]/self.inertias[n])
  fm,bf,bt,pm,dm=self.evaluate_viscous(um,vm,bm);u=s.u+dt*vm;v=s.v+dt*self.acceleration(fm);bnew={}
  for n,b in s.spheres.items():bnew[n]=dict(center=b['center']+dt*bm[n]['velocity'],velocity=b['velocity']+dt*bf[n]/self.bodies[n].mass,omega=b['omega']+dt*bt[n]/self.inertias[n])
  new=State(s.time+dt,u,v,bnew,*self.evaluate_viscous(u,v,bnew));increment=dt*dm['viscous_power_W'];self.dissipated_J+=increment;new.diagnostics['dissipated_energy_increment_J']=increment;new.diagnostics['dissipated_energy_J']=self.dissipated_J
  mid=State(s.time+.5*dt,um,vm,bm,fm,bf,bt,pm,dm);f,t=self.total_force_torque(mid);new.diagnostics['momentum_step_balance_Ns']=float(np.linalg.norm(self.momentum(new)-self.momentum(s)-dt*f));new.diagnostics['angular_step_balance_Nms']=float(np.linalg.norm(self.angular_momentum(new)-self.angular_momentum(s)-dt*t));return new
