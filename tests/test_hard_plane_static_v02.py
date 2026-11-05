import numpy as np
import pytest
import torch
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.hard_plane_static import StaticHardPlaneFEM


def fixture(attachment=False):
    p=falling_cube_payload(duration=.01);p['nodes']=p['nodes']-[0,0,.2]
    if attachment:
        w=(p['nodes'][:,2]>.05).astype(float)
        p['attachment_weights']=w[None];p['attachment_active']=np.ones((2,1))
        p['controller_targets']=np.broadcast_to(p['nodes'],(2,1,8,3)).copy()
    m=StaticHardPlaneFEM(p,device='cpu',field=False)
    theta=torch.zeros(m.i_field+1,dtype=torch.float64);theta[0]=np.log(.03)
    if attachment:theta[1]=np.log(2.)
    return m,theta


def check(m,theta,x,diag):
    E,r,_=m.evaluate(x,theta,0);k=m.kkt(x,r)
    assert k['stationarity_norm']<=1e-7
    assert k['maximum_penetration']<=1e-12
    assert k['minimum_reaction']>=0
    assert k['complementarity_max']<=1e-10
    assert diag['minJ']>.03 and not diag['posthoc_projection']
    assert torch.linalg.vector_norm(r.sum(0)-torch.tensor([0.,0.,k['reaction'].sum()],dtype=m.dtype))<1e-7


def test_gravity_supported_by_nonnegative_plane_reactions():
    m,t=fixture();x,_,d=m.solve(t,0);check(m,t,x,d)
    assert sum(d['normal_reactions'])==pytest.approx(float(t[0].exp()),abs=1e-7)
    assert np.count_nonzero(np.array(d['normal_reactions'])>0)>=3


def test_prescribed_downward_push_adds_compressive_reaction():
    m,t=fixture(True);m.targets[:,:, :,2]-=.05
    x,_,d=m.solve(t,0);check(m,t,x,d)
    assert sum(d['normal_reactions'])>float(t[0].exp())


def test_lifting_releases_plane_without_tension():
    m,t=fixture(True);supported,_,_=m.solve(t,0)
    m.targets[:,:,:,2]+=.4
    x,_,d=m.solve(t,0,initial=supported);check(m,t,x,d)
    assert float((x[:,2]-m.floor).min())>.1
    assert sum(d['normal_reactions'])==0.


def test_objective_gradient_tangent_and_ignored_soft_floor_slot():
    m,t=fixture();x=(m.rest+torch.tensor([0.,0.,.02],dtype=m.dtype)).requires_grad_()
    E,r,H=m.evaluate(x,t,0,True)
    grad=torch.autograd.grad(E,x,create_graph=True)[0]
    np.testing.assert_allclose(r.detach(),grad.detach(),atol=1e-12)
    auto=torch.autograd.functional.hessian(lambda q:m.evaluate(q.reshape_as(x),t,0)[0],x.flatten())
    np.testing.assert_allclose(H.detach(),auto.detach(),atol=1e-11)
    changed=t.clone();changed[m.i_floor]=30
    E2,r2,H2=m.evaluate(x,changed,0,True)
    assert torch.equal(E,E2)and torch.equal(r,r2)and torch.equal(H,H2)


def test_rejects_penetrating_initial_state_and_dynamics():
    m,t=fixture()
    with pytest.raises(ValueError,match='feasible'):m.solve(t,0,initial=m.rest-m.t([0.,0.,.01]))
    m.inertial_target=m.rest
    with pytest.raises(ValueError,match='Static'):m.evaluate(m.rest,t,0)
