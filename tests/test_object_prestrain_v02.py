"""Independent objectivity, mechanics and derivative tests for initial prestrain."""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.gaussian_boundary_flow_v02.object_prestrain import InitialPrestrainFEM,natural_neo_hookean


def payload():
    nodes=np.array(list(__import__('itertools').product((-.1,.1),repeat=3)),float)
    return {'nodes':nodes,'attachment_weights':np.ones((1,8)),
            'controller_targets':np.tile(nodes,(2,1,1,1)),
            'attachment_active':np.ones((2,1)),'floor':-.1}


def model_and_theta():
    m=InitialPrestrainFEM(payload(),device='cpu',field=False,contact=False,materials=1)
    theta=torch.zeros(m.parameter_count,dtype=torch.float64);theta[m.i_prestrain]=.7
    return m,theta


def test_zero_prestrain_reduces_to_existing_fem():
    m,theta=model_and_theta();theta[-1]=0
    old=ExternalFEM(payload(),device='cpu',field=False,contact=False,materials=1)
    x=m.rest@torch.diag(torch.tensor([1.03,.98,1.04],dtype=m.dtype))
    a=m.evaluate(x,theta,0,True);b=old.evaluate(x,theta[:-1],0,True)
    for left,right in zip(a,b):assert torch.allclose(left,right,atol=1e-12,rtol=1e-11)


def test_spatial_rigid_rotation_objectivity_and_piola_covariance():
    dtype=torch.float64;F=torch.tensor([[[1.1,.1,0.],[0.,.95,.04],[.01,0.,1.02]]],dtype=dtype)
    A=torch.diag(torch.tensor([1.,1.,1.2],dtype=dtype))[None]
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=dtype)
    mu=torch.tensor([.4],dtype=dtype);lam=torch.tensor([.8],dtype=dtype)
    W,P,_=natural_neo_hookean(F,A,mu,lam);Wr,Pr,_=natural_neo_hookean(R[None]@F,A,mu,lam)
    assert torch.allclose(W,Wr,atol=1e-14)
    assert torch.allclose(R[None]@P,Pr,atol=1e-14)


def test_bulk_has_zero_net_force_and_current_configuration_torque():
    m,theta=model_and_theta();x=m.rest@torch.tensor([[1.03,.05,0.],[0.,.98,.03],[0.,0.,1.02]],dtype=m.dtype)
    _,r,_=m.bulk_response(x,theta,m.natural_metric(theta))
    assert r.sum(0).abs().max()<1e-12
    torque=torch.linalg.cross(x,r).sum(0)
    assert torque.abs().max()<1e-12


def test_analytic_piola_and_tangent_match_autograd_and_finite_difference():
    dtype=torch.float64;F=torch.tensor([[[1.04,.02,0.],[.01,.97,.03],[0.,0.,1.08]]],dtype=dtype,requires_grad=True)
    A=torch.diag(torch.tensor([1.,1.,1.15],dtype=dtype))[None]
    mu=torch.tensor([.37],dtype=dtype);lam=torch.tensor([.82],dtype=dtype)
    W,P,C=natural_neo_hookean(F,A,mu,lam,tangent=True)
    autoP=torch.autograd.grad(W.sum(),F,create_graph=True)[0]
    autoC=torch.autograd.functional.jacobian(lambda f:natural_neo_hookean(f,A,mu,lam)[1],F)[0,:,:,0]
    assert torch.allclose(P,autoP,atol=1e-12,rtol=1e-11)
    assert torch.allclose(C[0],autoC,atol=1e-12,rtol=1e-11)
    direction=torch.tensor([[[.2,-.1,.3],[.1,.4,-.2],[-.2,.1,.3]]],dtype=dtype);eps=1e-6
    plus=natural_neo_hookean(F.detach()+eps*direction,A,mu,lam)[1]
    minus=natural_neo_hookean(F.detach()-eps*direction,A,mu,lam)[1]
    analytic=torch.einsum('eijkl,ekl->eij',C,direction)
    assert torch.allclose(analytic,(plus-minus)/(2*eps),atol=1e-9,rtol=1e-7)


def test_fem_residual_and_tangent_match_autograd():
    m,theta=model_and_theta();x=(m.rest@torch.diag(torch.tensor([1.02,.99,1.01],dtype=m.dtype))).requires_grad_(True)
    A=m.natural_metric(theta);E,r,H=m.bulk_response(x,theta,A,tangent=True)
    auto=torch.autograd.grad(E,x,create_graph=True)[0]
    autoH=torch.autograd.functional.jacobian(lambda q:m.bulk_response(q,theta,A)[1],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(r,auto,atol=1e-12,rtol=1e-11)
    assert torch.allclose(H,autoH,atol=1e-12,rtol=1e-11)
    assert torch.allclose(H,H.T,atol=1e-12)


def test_prestrain_derivative_matches_finite_difference():
    m,theta=model_and_theta();theta.requires_grad_(True);x=m.rest
    E,_,_=m.evaluate(x,theta,0);g=torch.autograd.grad(E,theta)[0][-1]
    eps=1e-6;delta=torch.zeros_like(theta);delta[-1]=eps
    plus=m.evaluate(x,theta.detach()+delta,0)[0];minus=m.evaluate(x,theta.detach()-delta,0)[0]
    assert torch.allclose(g,(plus-minus)/(2*eps),atol=1e-9,rtol=1e-6)


def test_freefall_net_gravity_unchanged_without_supports():
    m,theta=model_and_theta();theta[1]=-torch.inf;theta[m.i_floor]=-torch.inf
    m.active[:]=0
    # At rest the inherited positional gauge is exactly zero; a bulk prestrain
    # may redistribute internal forces but cannot change their resultant.
    _,residual,_=m.evaluate(m.rest,theta,0)
    expected=m.t([0.,0.,torch.exp(theta[0]).item()])
    assert torch.allclose(residual.sum(0),expected,atol=1e-12)
    zero=theta.clone();zero[-1]=0
    _,r0,_=m.evaluate(m.rest,zero,0)
    assert torch.allclose(residual.sum(0),r0.sum(0),atol=1e-12)
    baseline=ExternalFEM(payload(),device='cpu',field=False,contact=False,materials=1)
    assert torch.equal(m.mass,baseline.mass)


def test_prestrain_cannot_replace_missing_initial_support_reaction():
    m,theta=model_and_theta()
    # Compliant anchors are active but exactly unextended; the penalty floor
    # has no penetration. Internal eigenstress cannot provide a net body load.
    for amplitude in (-2.,0.,2.):
        theta[-1]=amplitude
        _,r,_=m.evaluate(m.rest,theta,0)
        assert torch.allclose(r.sum(0),m.t([0.,0.,1.]),atol=1e-12)


def test_released_translated_state_preserves_existing_external_resultant():
    m,theta=model_and_theta();theta[1]=-torch.inf;theta[m.i_floor]=-torch.inf;m.active[:]=0
    x=m.rest+m.t([.1,-.2,.3])
    _,r,_=m.evaluate(x,theta,0)
    zero=theta.clone();zero[-1]=0
    _,r0,_=m.evaluate(x,zero,0)
    # The pre-existing weak positional gauge is retained, so this compares the
    # exact existing external resultant rather than claiming ideal freefall.
    assert torch.allclose(r.sum(0),r0.sum(0),atol=1e-12)
