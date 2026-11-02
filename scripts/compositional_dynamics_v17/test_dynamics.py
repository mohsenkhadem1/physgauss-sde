import numpy as np
from scipy.spatial.transform import Rotation
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet,SphereBody
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidSphere


def model(**kwargs):return DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=1,order=3,support_mode='free',surface_order=4,**kwargs)


def test_mass_exact_uniform_body_moments_and_ballistics():
 m=model(gravity=(.2,-.3,-9.81));v=VelocityVerlet(m);X=m.rest;center=np.average(X,axis=0,weights=np.asarray(v.M.sum(1)).ravel());A=X-center
 assert abs(v.M.sum()-625*.12*.06*.06)<1e-14
 second=A.T@(v.M@A);np.testing.assert_allclose(second,np.diag(m.total_mass*np.square(m.geometry.dimensions)/12),atol=2e-17)
 initial=np.tile([.1,.2,-.1],(len(X),1));s=v.initial_state(v=initial);E=v.energy(s)
 for _ in range(20):s=v.step(s,.0001)
 np.testing.assert_allclose(s.u,s.time*initial+.5*s.time**2*m.gravity,atol=5e-14)
 np.testing.assert_allclose(s.v,initial+s.time*m.gravity,atol=1e-10)
 assert abs(v.energy(s)-E)<1e-12
 assert s.diagnostics['momentum_step_balance_Ns']<1e-13


def test_manufactured_harmonic_oscillator_second_order_without_restitution():
 errors=[]
 for dt in [.002,.001]:
  m=model();v=VelocityVerlet(m);omega=20.
  def evaluate(u,tangent=False):
   U=.5*omega**2*np.sum(u*(v.M@u));return dict(residual=omega**2*(v.M@u),potential_J=U,elastic_J=U,contact_J=0.,spring_J=0.,J=np.ones(1),max_sampled_penetration_m=0.)
  m.evaluate=evaluate;u=np.zeros_like(m.rest);u[:,0]=.001;s=v.initial_state(u=u);E=v.energy(s)
  for _ in range(round(.2/dt)):s=v.step(s,dt)
  errors.append(np.max(abs(s.u[:,0]-.001*np.cos(omega*.2))))
  assert abs(v.energy(s)-E)/E<.001
 assert 3.9<errors[0]/errors[1]<4.1


def test_coupled_sphere_impulse_angular_momentum_and_objectivity():
 R=Rotation.from_rotvec([.2,-.3,.4]).as_matrix();offset=np.array([.13,.04,-.05]);cases=[]
 for Q,t in [(np.eye(3),np.zeros(3)),(R,offset)]:
  center=np.array([.06,0,.049]);c=RigidSphere(center=Q@center+t,radius=.02,penalty_N_m3=1e6,name='ball')
  m=model(colliders=(c,));body=SphereBody('ball',.05,Q@center+t,Q@np.array([.03,0,-.2]),Q@np.array([.2,.1,.3]));v=VelocityVerlet(m,bodies=[body]);u=m.rest@Q.T+t-m.rest;s=v.initial_state(u=u);P=v.momentum(s);L=v.angular_momentum(s)
  for _ in range(8):s=v.step(s,1e-5)
  np.testing.assert_allclose(v.momentum(s),P,atol=2e-12);np.testing.assert_allclose(v.angular_momentum(s),L,atol=2e-12)
  assert s.diagnostics['momentum_step_balance_Ns']<1e-13
  cases.append((m,s))
 np.testing.assert_allclose(cases[1][0].rest+cases[1][1].u,(cases[0][0].rest+cases[0][1].u)@R.T+offset,atol=2e-12)
 np.testing.assert_allclose(cases[1][1].v,cases[0][1].v@R.T,atol=2e-10)


def test_stable_nh_control_matches_existing_independent_numpy_energy_and_stress():
 import torch
 from scripts.compositional_dynamics_v17.materials import StableNeoHookean
 from experiments.campaign_newton_stable_material import stable_material
 F=np.array([[[1.2,.15,0],[0,.9,.1],[0,0,1.05]],np.eye(3)])
 x=torch.tensor(F,dtype=torch.float64,requires_grad=True);law=StableNeoHookean();w=law(x,None);p=torch.autograd.grad(w.sum(),x)[0].detach().numpy();ref,stress,_=stable_material(F,12000.,.35,False)
 np.testing.assert_allclose(w.detach().numpy(),ref,atol=1e-11);np.testing.assert_allclose(p,stress,atol=1e-10)
