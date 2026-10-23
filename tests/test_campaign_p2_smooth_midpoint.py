import numpy as np
import pytest
import torch
from experiments.campaign_p2_transient import P2State
from experiments.campaign_p2_smooth_midpoint import midpoint_step,smooth_bump


class Toy:
 def __init__(self,nonlinear=0.,fixed=False):
  self.rest=torch.zeros((2,3),dtype=torch.float64);self.mask=torch.zeros_like(self.rest,dtype=torch.bool)
  if fixed:self.mask[0]=True
  self.M=torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64);self.K=torch.tensor([[4.,-1.],[-1.,3.]],dtype=torch.float64);self.beta=nonlinear;self.minimum_J=.1
 def mass(self,v):return self.M@v
 def evaluate(self,u):return dict(energy=.5*(u*(self.K@u)).sum()+self.beta*(u**4).sum()/4,gradient=self.K@u+self.beta*u**3,minimum_J=torch.tensor(1.))
 def solve(self,force,initial,target,effective_dt,force_tolerance):
  x=initial.clone();free=np.flatnonzero(~self.mask.numpy().ravel());M=torch.kron(self.M,torch.eye(3,dtype=torch.float64));K=torch.kron(self.K,torch.eye(3,dtype=torch.float64))
  for n in range(15):
   c=self.evaluate(x);r=c['gradient']-force+self.mass(x-target)/effective_dt**2
   if float(torch.linalg.vector_norm(r[~self.mask]))<=force_tolerance:break
   H=K+M/effective_dt**2+torch.diag((3*self.beta*x**2).ravel());dx=torch.linalg.solve(H[free][:,free],-r.ravel()[free]);x.ravel()[free]+=dx
  else:raise RuntimeError('Toy Newton failed')
  return dict(displacement=x,cache=c,residual=r.masked_fill(self.mask,0.),iterations=n,history=[])


def test_full_mass_linear_oscillator_matches_independent_first_order_cayley_and_energy():
 m=Toy();u=torch.tensor([[.1,.02,0.],[-.04,0.,.03]],dtype=torch.float64);v=torch.tensor([[.03,0.,0.],[0.,-.02,0.]],dtype=torch.float64);F=torch.ones_like(u)*.07;h=.03
 new,row=midpoint_step(m,P2State(u,v),h,F)
 M=np.kron(m.M.numpy(),np.eye(3));K=np.kron(m.K.numpy(),np.eye(3));A=np.block([[np.zeros((6,6)),np.eye(6)],[-np.linalg.solve(M,K),np.zeros((6,6))]]);b=np.r_[np.zeros(6),np.linalg.solve(M,F.numpy().ravel())];y=np.r_[u.numpy().ravel(),v.numpy().ravel()];expected=np.linalg.solve(np.eye(12)-h*A/2,(np.eye(12)+h*A/2)@y+h*b)
 np.testing.assert_allclose(np.r_[new.u.numpy().ravel(),new.v.numpy().ravel()],expected,rtol=1e-13,atol=1e-14)
 assert abs(row['nonlinear_midpoint_internal_energy_defect_J'])<1e-15 and abs(row['total_energy_minus_midpoint_work_J'])<1e-14


def test_nonlinear_work_identity_and_energy_gradient_finite_difference():
 m=Toy(nonlinear=20.);u=torch.tensor([[.1,.02,0.],[-.04,0.,.03]],dtype=torch.float64);v=torch.ones_like(u)*.05;F=torch.ones_like(u)*.1
 new,row=midpoint_step(m,P2State(u,v),.05,F)
 assert abs(row['nonlinear_midpoint_internal_energy_defect_J'])>1e-10
 assert abs(row['ledger_identity_defect_J'])<1e-14
 d=torch.tensor(np.random.default_rng(1).normal(size=(2,3)),dtype=torch.float64);eps=1e-6
 fd=(m.evaluate(u+eps*d)['energy']-m.evaluate(u-eps*d)['energy'])/(2*eps)
 torch.testing.assert_close(fd,(m.evaluate(u)['gradient']*d).sum(),rtol=1e-8,atol=1e-10)


def test_exact_constraints_reactions_and_public_smooth_pulse():
 m=Toy(fixed=True);zero=torch.zeros_like(m.rest);F=torch.zeros_like(zero);F[1,2]=1.
 new,row=midpoint_step(m,P2State(zero,zero),.02,F)
 assert torch.equal(new.u[0],zero[0]) and torch.equal(new.v[0],zero[0])
 assert abs(row['fixed_reaction_resultant_N'][2])>0 # M_cf and K_cf retained
 assert smooth_bump(0)==smooth_bump(1)==smooth_bump(-1)==smooth_bump(2)==0 and smooth_bump(.5)==1
 assert smooth_bump(1e-5)<3e-18


def test_endpoint_guard_is_independent_of_midpoint_guard():
 m=Toy();original=m.evaluate
 def evaluate(u):
  c=original(u)
  if u.max()>.00015:c['minimum_J']=torch.tensor(-.1)
  return c
 m.evaluate=evaluate;z=torch.zeros_like(m.rest)
 with pytest.raises(RuntimeError,match='positive-J'):midpoint_step(m,P2State(z,z),.04,torch.ones_like(z))


def test_position_AND_velocity_second_order_against_independent_exact_flow():
 from scipy.linalg import expm
 m=Toy();u=torch.tensor([[.1,.02,0.],[-.04,0.,.03]],dtype=torch.float64);v=torch.zeros_like(u);F=torch.ones_like(u)*.07;T=.2
 M=np.kron(m.M.numpy(),np.eye(3));K=np.kron(m.K.numpy(),np.eye(3));A=np.block([[np.zeros((6,6)),np.eye(6)],[-np.linalg.solve(M,K),np.zeros((6,6))]]);equilibrium=np.r_[np.linalg.solve(K,F.numpy().ravel()),np.zeros(6)];exact=equilibrium+expm(T*A)@(np.r_[u.numpy().ravel(),v.numpy().ravel()]-equilibrium);errors=[]
 for steps in [8,16,32]:
  state=P2State(u.clone(),v.clone())
  for _ in range(steps):state,_=midpoint_step(m,state,T/steps,F)
  errors.append([np.linalg.norm(state.u.numpy().ravel()-exact[:6]),np.linalg.norm(state.v.numpy().ravel()-exact[6:])])
 errors=np.array(errors);assert np.all(errors[:-1]/errors[1:]>3.9)
