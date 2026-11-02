import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from scripts.compositional_neural_mechanics.contact_mechanics import (
    RigidPlane,RigidSphere,normal_contact_response,ContactQ2Mechanics)


def tensor(x):return torch.tensor(x,dtype=torch.float64)


@pytest.mark.parametrize('collider',[RigidPlane(point=(0,0,.01)),RigidSphere(radius=.04)])
def test_kernel_objectivity_tangent_and_separation(collider):
    x=tensor([[.01,.005,-.012],[.09,.02,.08]])
    W,p,C,gap=normal_contact_response(collider,x,tangent=True)
    assert W[0]>0 and W[1]==0 and torch.count_nonzero(p[1])==0
    R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix();t=np.array([.2,-.1,.03]);Rt=tensor(R)
    Wr,pr,Cr,_=normal_contact_response(collider.transformed(R,t),x@Rt.T+tensor(t),tangent=True)
    torch.testing.assert_close(W,Wr,atol=1e-10,rtol=1e-11)
    torch.testing.assert_close(pr,p@Rt.T,atol=1e-8,rtol=1e-11)
    torch.testing.assert_close(Cr,Rt@C@Rt.T,atol=1e-7,rtol=1e-11)
    d=tensor([[.3,-.1,.4],[-.1,.2,.3]]);h=1e-7
    _,pp,_,_=normal_contact_response(collider,x+h*d)
    _,pm,_,_=normal_contact_response(collider,x-h*d)
    torch.testing.assert_close((pp-pm)/(2*h),torch.einsum('bij,bj->bi',C,d),atol=1e-4,rtol=1e-8)
    assert torch.all((p*(x-collider.origin(x))).sum(-1)<=0) if isinstance(collider,RigidSphere) else p[0,2]<0


def test_reference_surface_area_and_contact_scatter_derivatives():
    model=ContactQ2Mechanics(n=1,colliders=(RigidPlane(point=(0,0,-.039)),))
    assert abs(sum(p['weights'].sum() for p in model.contact_patches)-2*(.24*.08*2+.08**2))<1e-14
    u=tensor(np.zeros_like(model.rest));u.requires_grad_(True)
    U,r,H,records=model.contact_response(u,tangent=True,create_graph=True)
    torch.testing.assert_close(torch.autograd.grad(U,u,retain_graph=True)[0],r,atol=1e-12,rtol=1e-12)
    direction=tensor(np.random.default_rng(42).normal(size=model.rest.shape))
    hv=torch.autograd.grad((r*direction).sum(),u)[0].detach().numpy().ravel()
    np.testing.assert_allclose(hv,H@direction.numpy().ravel(),atol=1e-9,rtol=1e-12)
    rigid=records[0]
    torch.testing.assert_close(r.sum(0),rigid['force_on_rigid_N'],atol=1e-12,rtol=1e-12)
    world_moment=torch.linalg.cross(tensor(model.rest)+u,r,dim=-1).sum(0)
    expected=rigid['torque_on_rigid_Nm']+torch.linalg.cross(rigid['origin_world_m'],rigid['force_on_rigid_N'])
    torch.testing.assert_close(world_moment,expected,atol=1e-12,rtol=1e-12)
    # Prescribed rigid translation work conjugacy; the geometry parameter keeps gradients.
    origin=tensor([0,0,-.039]);origin.requires_grad_(True)
    model.colliders=(RigidPlane(point=origin),)
    Ut,_,_,rec=model.contact_response(u,create_graph=True)
    torch.testing.assert_close(-torch.autograd.grad(Ut,origin)[0],rec[0]['force_on_rigid_N'],atol=1e-12,rtol=1e-12)


def test_gravity_contact_equilibrium_and_full_tangent():
    model=ContactQ2Mechanics(n=1,gravity=(0,0,-9.81),colliders=(RigidPlane(point=(0,0,-.04001)),))
    state=model.solve(force_tolerance_N=1e-8)
    w=model.reaction_wrenches(state);c=w['contact_reactions'][0]
    assert np.linalg.norm(state['residual'])<1e-8
    assert c['active_quadrature_samples']>0 and c['force_on_rigid_N'][2]<0
    assert 0<c['max_sampled_penetration_m']<.001
    assert w['force_balance_N']<1e-7 and w['world_torque_balance_Nm']<1e-7
    u=tensor(state['displacement_local_m']);u.requires_grad_(True)
    r=model.torch_residual(u);np.testing.assert_allclose(r.detach(),state['residual'],atol=1e-10)
    d=tensor(np.random.default_rng(7).normal(size=u.shape))
    hv=torch.autograd.grad((r*d).sum(),u)[0].detach().numpy().ravel()
    np.testing.assert_allclose(hv,model.evaluate(state['displacement_local_m'],True)['tangent']@d.numpy().ravel(),atol=1e-8,rtol=1e-9)


def test_invalid_and_open_contacts():
    with pytest.raises(ValueError):normal_contact_response(RigidSphere(),tensor([[0,0,0]]))
    with pytest.raises(ValueError):normal_contact_response(RigidPlane(normal=(0,0,0)),tensor([[0,0,.1]]))
    with pytest.raises(ValueError):normal_contact_response(RigidPlane(),tensor([[0,0,float('nan')]]))
    model=ContactQ2Mechanics(n=1,colliders=(RigidPlane(point=(0,0,-.1)),))
    U,r,H,rec=model.contact_response(tensor(np.zeros_like(model.rest)),tangent=True)
    assert U==0 and torch.count_nonzero(r)==0 and H.count_nonzero()==0 and rec[0]['active_quadrature_samples']==0


def test_prescribed_rotation_work_and_sphere_equilibrium():
    origin=np.array([.12,0,-.039]);model=ContactQ2Mechanics(n=1,colliders=(RigidPlane(point=origin),))
    u=tensor(np.zeros_like(model.rest));u[:,2]=tensor(.002*(model.rest[:,0]-.12)/.12)
    _,_,_,records=model.contact_response(u)
    torque=records[0]['torque_on_rigid_Nm'].detach().numpy()
    h=1e-6;energies=[]
    for angle in (h,-h):
        normal=Rotation.from_rotvec([0,angle,0]).apply([0,0,1])
        model.colliders=(RigidPlane(point=origin,normal=normal),)
        energies.append(float(model.contact_response(u)[0]))
    assert abs((energies[0]-energies[1])/(2*h)+torque[1])<1e-7
    sphere=ContactQ2Mechanics(n=1,colliders=(RigidSphere(center=(.12,0,-.07),radius=.045),))
    state=sphere.solve(force_tolerance_N=1e-8);w=sphere.reaction_wrenches(state)
    assert state['J'].min()>.1 and w['force_balance_N']<1e-7 and w['world_torque_balance_Nm']<1e-7
    assert w['contact_reactions'][0]['active_quadrature_samples']>0
    assert np.linalg.norm(w['contact_reactions'][0]['torque_on_rigid_Nm'])<1e-12


def test_independent_dense_check_detects_missed_small_patch():
    from scripts.compositional_neural_mechanics.contact_mechanics import dense_surface_gap_check
    model=ContactQ2Mechanics(n=1,surface_order=4,colliders=(RigidSphere(center=(.12,0,-.07),radius=.031),))
    report=dense_surface_gap_check(model,np.zeros_like(model.rest),17)[0]
    assert report['missed_contact'] and abs(report['max_dense_penetration_m']-.001)<1e-14
    assert report['minimum_material_coordinate']==[.5,.5,0.]
