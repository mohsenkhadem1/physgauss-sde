import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts.compositional_neural_mechanics.mechanics import (Q2Mechanics,GeometrySpec,AttachmentSpec,AnalyticNeoHookean,constitutive_response)


def test_reference_objectivity_and_tangent_derivative():
    law=AnalyticNeoHookean();z=torch.tensor([[np.log(12000.)]],dtype=torch.float64)
    I=torch.eye(3,dtype=torch.float64)[None];W,P,A=constitutive_response(law,I,z,tangent=True)
    assert abs(W.item())<1e-12 and P.abs().max()<1e-10
    F=I+torch.tensor([[[.05,.02,0.],[0.,-.02,.03],[0.,0.,.01]]],dtype=torch.float64)
    R=torch.tensor(Rotation.from_rotvec([.3,-.2,.1]).as_matrix(),dtype=torch.float64)
    W,P,A=constitutive_response(law,F,z,tangent=True);Wr,Pr,_=constitutive_response(law,R@F,z)
    torch.testing.assert_close(W,Wr,atol=1e-10,rtol=1e-10);torch.testing.assert_close(Pr,R@P,atol=1e-9,rtol=1e-10)
    h=1e-6;D=torch.arange(9,dtype=torch.float64).reshape(1,3,3)/10
    _,plus,_=constitutive_response(law,F+h*D,z);_,minus,_=constitutive_response(law,F-h*D,z)
    torch.testing.assert_close((plus-minus)/(2*h),torch.einsum('biJkL,bkL->biJ',A,D),atol=1e-5,rtol=1e-7)


def test_curved_reference_affine_patch_and_rigid_invariance():
    model=Q2Mechanics(GeometrySpec((.2,.06,.09),taper=.3,arch=.012),n=1)
    assert abs(model.volumes.sum()-.2*.06*.09*(1+.3**2/3))<1e-12
    R=Rotation.from_rotvec([.2,.1,-.1]).as_matrix();u=model.rest@R.T+np.array([.02,-.01,.03])-model.rest
    U,g,H,J=model.elastic(model.rest+u,True)
    assert abs(U)<1e-11 and np.linalg.norm(g)<1e-9 and np.max(abs(J-1))<1e-12
    q=np.array([[.1,.4,.7],[.8,.9,.2]]);expected=model.geometry.map(q)@R.T+np.array([.02,-.01,.03])-model.geometry.map(q)
    np.testing.assert_allclose(model.sample_material(u,q),expected,atol=1e-14)
    np.testing.assert_allclose(model.sample_material(u,q,True),np.broadcast_to(R-np.eye(3),(2,3,3)),atol=1e-13)


def test_position_equilibrium_and_torch_residual():
    model=Q2Mechanics(n=1);poses=np.zeros((2,6));poses[1,0]=.002;model.set_controls(poses)
    state=model.solve(force_tolerance_N=1e-9);w=model.reaction_wrenches(state)
    assert np.linalg.norm(state['residual'])<1e-9 and w['force_balance_N']<1e-8 and state['J'].min()>.1
    u=torch.tensor(state['displacement_local_m'],dtype=torch.float64,requires_grad=True)
    r=model.torch_residual(u);np.testing.assert_allclose(r.detach(),state['residual'],atol=1e-10)
    direction=torch.randn_like(u);grad=torch.autograd.grad((r*direction).sum(),u)[0]
    H=model.evaluate(state['displacement_local_m'],True)['tangent']
    np.testing.assert_allclose(grad.detach().numpy().ravel(),H.T@direction.detach().numpy().ravel(),atol=1e-8,rtol=1e-8)


def test_quadrature_interface_equals_linear_robin_and_torque():
    def energy(delta,z):return .5*torch.exp(z[...,0])*(delta*delta).sum(-1)
    def field(q):return torch.full((*q.shape[:-1],1),np.log(6000.),dtype=q.dtype,device=q.device)
    geometry=GeometrySpec(taper=.2,arch=.01)
    linear=Q2Mechanics(geometry,n=1)
    nonlinear=Q2Mechanics(geometry,n=1,interface_energy=energy,interface_latent_fields=(field,field))
    pose=np.zeros((2,6));pose[1,5]=.05;linear.set_controls(pose);nonlinear.set_controls(pose)
    u=np.random.default_rng(3).normal(0,.0001,linear.rest.shape)
    a=linear.evaluate(u,True);b=nonlinear.evaluate(u,True)
    np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-12);np.testing.assert_allclose(a['tangent'].toarray(),b['tangent'].toarray(),atol=1e-10)
    assert abs(a['potential_J']-b['potential_J'])<1e-13
    wa=linear.reaction_wrenches({'displacement_local_m':u});wb=nonlinear.reaction_wrenches({'displacement_local_m':u})
    for key in ('forces_N','torques_Nm'):np.testing.assert_allclose(wa[key],wb[key],atol=1e-12)
    h=1e-6;plus=pose.copy();minus=pose.copy();plus[1,5]+=h;minus[1,5]-=h
    nonlinear.set_controls(plus);ep=nonlinear.evaluate(u)['potential_J'];nonlinear.set_controls(minus);em=nonlinear.evaluate(u)['potential_J']
    assert abs((ep-em)/(2*h)+wb['torques_Nm'][1,2])<1e-9


def test_analytic_oracle_matches_existing_q2_operator():
    from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
    old=SmoothMaterialFixture(2,12000.,6000.,6000.,'parallel_lower',device='cpu',gravity=(0,0,0))
    new=Q2Mechanics(n=2);pose=np.zeros((2,6));pose[1,3]=.1
    old.set_controls(pose);new.set_controls(pose)
    np.testing.assert_allclose(old.rest,new.rest,atol=1e-16)
    u=np.random.default_rng(2).normal(0,.00001,new.rest.shape)
    a=old.evaluate(u,True);b=new.evaluate(u,True)
    assert abs(a['potential_J']-b['potential_J'])<1e-12
    np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-11,rtol=1e-10)
    np.testing.assert_allclose(a['tangent'].toarray(),b['tangent'].toarray(),atol=1e-9,rtol=1e-9)


def test_differentiable_footprint_defaults_preserve_cached_operator():
    def energy(delta,z):return .5*torch.exp(z[...,0])*(delta*delta).sum(-1)
    def field(q):return q.new_full((*q.shape[:-1],1),np.log(6000.))
    settings=dict(n=1,interface_energy=energy,interface_latent_fields=(field,field))
    cached=Q2Mechanics(**settings)
    parameters=torch.tensor([[.5,.1875,np.log(.375),np.log(.375)]]*2,dtype=torch.float64,requires_grad=True)
    live=Q2Mechanics(**settings,interface_geometry_parameters=parameters)
    pose=np.zeros((2,6));pose[1,5]=.1;cached.set_controls(pose);live.set_controls(pose)
    u=np.random.default_rng(4).normal(0,.0001,cached.rest.shape)
    a=cached.evaluate(u,True);b=live.evaluate(u,True)
    np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-12);np.testing.assert_allclose(a['tangent'].toarray(),b['tangent'].toarray(),atol=1e-11)
    r=live.torch_residual(torch.tensor(u,dtype=torch.float64,requires_grad=True));g=torch.autograd.grad(r.square().sum(),parameters)[0]
    assert torch.isfinite(g).all() and g.abs().max()>1e-6
