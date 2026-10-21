"""Independent tiny mechanics checks; no cohort values or simulator calls."""
import numpy as np
import pytest
import torch
from experiments.campaign_nonlinear_gaussian_reference import (
    NonlinearGaussianReference, StepFailure, tensor, tiny_model, transient)
from experiments.campaign_compliant_gaussian_linear import (
    LinearCompliantSystem, tiny_geometry, candidate_patches)
from experiments.campaign_nonlinear_fem import InvalidConfiguration
from experiments import campaign_boundary_gaussians as boundary

torch.set_num_threads(2)


def rotation():
    n=np.array([1.,2.,-1.]);n/=np.linalg.norm(n);a=1.9
    W=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.eye(3)+np.sin(a)*W+(1-np.cos(a))*(W@W)


def test_finite_rigid_rotation_and_rotated_reference():
    m=tiny_model();Q=tensor(rotation());shift=tensor([.3,-.2,.1])
    x=m.rest@Q.T+shift
    U,J=m.elastic(x)
    assert abs(float(U))<2e-14
    torch.testing.assert_close(J,torch.ones_like(J),rtol=0,atol=3e-15)
    a=x.clone().requires_grad_(True)
    assert float(torch.autograd.grad(m.elastic(a)[0],a)[0].abs().max())<1e-12
    # Rotate the complete physical reference/interaction tensors, not only x.
    X,tets,faces=tiny_geometry();patch,K,D=candidate_patches(rotation(),shift.numpy())
    load=[boundary.GaussianPatch(p.center_material_m,p.covariance_material_m2,'known_hardware') for p in patch]
    other=NonlinearGaussianReference(x.numpy(),tets,faces,young=5000,poisson=.35,density=625,
        support_patches=patch,stiffness=K,damping=D,load_patches=load)
    f=m.known_force([[.003,.001,0],[0,-.001,0]])
    targets=m.rest[None].repeat(2,1,1); targets2=targets@Q.T+shift
    y,v,row=m.step(m.rest,torch.zeros_like(m.rest),targets,targets,f,.02)
    yr,vr,rr=other.step(x,torch.zeros_like(x),targets2,targets2,f@Q.T,.02)
    torch.testing.assert_close(yr,y@Q.T+shift,rtol=0,atol=5e-15)
    torch.testing.assert_close(vr,v@Q.T,rtol=0,atol=3e-13)


def test_energy_gradient_independent_first_piola_and_finite_difference():
    m=tiny_model();F=tensor([[1.1,.07,0],[.02,.95,.04],[0,0,1.03]])
    x=(m.rest@F.T+tensor([.03,-.01,.02])).requires_grad_(True)
    grad=torch.autograd.grad(m.elastic(x)[0],x)[0]
    # P=mu(F-F^-T)+lambda log(detF) F^-T; independent matrix derivative.
    mu=m.young/(2*(1+m.poisson));lam=m.young*m.poisson/((1+m.poisson)*(1-2*m.poisson))
    invT=torch.linalg.inv(F).T;P=mu*(F-invT)+lam*torch.log(torch.linalg.det(F))*invT
    expected=torch.zeros_like(x)
    for e,ids in enumerate(m.tets): expected[ids]+=m.volumes[e]*(m.gradients[e]@P.T)
    torch.testing.assert_close(grad,expected,rtol=1e-12,atol=1e-13)
    # Independent public spring law also agrees with the total potential gradient.
    target=m.rest[None].repeat(2,1,1);zero=torch.zeros_like(x)
    response=m.fixture.response(x,zero,target,torch.zeros_like(target))
    total_grad=torch.autograd.grad(m.elastic(x)[0]+response['energy'],x)[0]
    torch.testing.assert_close(total_grad,expected-response['forces'],rtol=1e-12,atol=1e-13)
    direction=tensor(np.random.default_rng(2).normal(size=x.shape));eps=1e-7
    finite=(m.elastic(x.detach()+eps*direction)[0]-m.elastic(x.detach()-eps*direction)[0])/(2*eps)
    assert float(finite)==pytest.approx(float((grad*direction).sum()),rel=1e-8,abs=1e-9)


def test_tangent_small_force_limit_against_linear_and_known_controls():
    m=tiny_model();targets=m.rest[None].repeat(2,1,1)
    H=torch.autograd.functional.hessian(lambda u:m.elastic(m.rest+u.reshape_as(m.rest))[0],torch.zeros(m.rest.numel(),dtype=torch.float64))
    torch.testing.assert_close(H,m.K0,rtol=1e-12,atol=2e-12)
    linear=LinearCompliantSystem(m.K0,m.mass,m.fixture)
    f=m.known_force([[1e-7,-2e-7,1e-7],[0.,0.,0.]])
    x,v,row=m.step(m.rest,torch.zeros_like(m.rest),targets,targets,f,.03,absolute_tolerance_N=1e-14)
    ulin,_,_=linear.step(torch.zeros_like(m.rest),torch.zeros_like(m.rest),torch.zeros_like(targets),torch.zeros_like(targets),f,.03)
    torch.testing.assert_close(x-m.rest,ulin,rtol=2e-7,atol=2e-16)
    zero,_,_=m.step(m.rest,torch.zeros_like(m.rest),targets,targets,torch.zeros_like(f),.03)
    assert torch.equal(zero,m.rest) and float((x-zero).abs().max())>1e-10
    totals=tensor([[.1,.2,.3],[-.04,.01,0.]])
    gravity=tensor([.2,-.1,.3]);force=m.known_force(totals,gravity=gravity)
    torch.testing.assert_close(force.sum(0),totals.sum(0)+m.kernels.total_mass*gravity,rtol=0,atol=64*np.finfo(float).eps*float(totals.abs().sum()+m.kernels.total_mass*gravity.abs().sum()))
    assert float(m.mass.reshape(-1,3)[:,0].sum())==pytest.approx(float(m.kernels.total_mass),rel=1e-15)


def test_stationarity_dissipation_energy_identity_and_moving_exact_constraints():
    mask=np.zeros((5,3),dtype=bool);mask[0]=True;m=tiny_model(mask=mask)
    targets=m.rest[None].repeat(2,1,1); x=m.rest+tensor([.002,-.001,.0005]);v=torch.zeros_like(x)
    oldenergy=float(m.energy(x,v,targets)); totals=[]
    for k in range(12):
        x,v,row=m.step(x,v,targets,targets,torch.zeros_like(x),.01,constraint_positions=m.rest)
        assert row['force_residual_N']<=row['threshold_N']*1.0001
        assert row['constraint_max_error_m']==0 and torch.equal(x[0],m.rest[0])
        assert row['damping_dissipation_J']>=0
        assert abs(row['balance_error_J']-row['free_residual_work_J'])<2e-15
        energy=float(m.energy(x,v,targets))
        # The first imposed rest pin moves from its initially displaced position:
        # account for its reaction work before checking passivity.
        assert energy-oldenergy-row['reaction_work_J']<1e-12
        oldenergy=energy;totals.append(row)
    # Both moving soft targets and a moving exact pin have work in the ledger.
    targetnew=targets+tensor([.0001,.00002,0]); prescribed=m.rest+tensor([.00005,0,0])
    x,v,row=m.step(x,v,targets,targetnew,torch.zeros_like(x),.01,constraint_positions=prescribed)
    assert row['target_work_J']!=0 and row['reaction_work_J']!=0
    assert abs(row['balance_error_J']-row['free_residual_work_J'])<2e-15


def test_positive_J_explicit_failure_and_compatible_covariance_mass():
    m=tiny_model();F=tensor([[1.2,.1,0],[0,.9,.04],[0,0,1.1]])
    x=m.rest@F.T+tensor([.1,.2,.3]);state=m.kernels.transport(x)
    torch.testing.assert_close(state.deformation_gradients,F.expand(2,3,3),rtol=0,atol=1e-15)
    torch.testing.assert_close(state.covariances,F@m.kernels.reference_covariances@F.T,rtol=0,atol=3e-18)
    torch.testing.assert_close(state.bulk_density*state.volumes,m.kernels.masses,rtol=0,atol=3e-17)
    with pytest.raises(InvalidConfiguration): m.elastic(m.rest*tensor([-1.,1.,1.]))
    original=m.rest.clone();targets=m.rest[None].repeat(2,1,1)
    with pytest.raises(StepFailure) as failure:
        m.step(original,torch.zeros_like(original),targets,targets,m.known_force([[1.,0,0],[0,0,0]]),.05,maximum_iterations=1)
    assert len(failure.value.history)==2
    assert torch.equal(original,m.rest) # no failed-state mutation or replacement


def test_same_continuous_load_time_refinement():
    a,_=transient(tiny_model(),60);b,_=transient(tiny_model(),120);c,rows=transient(tiny_model(),240)
    ab=float(torch.sqrt(((a-b[::2])**2).mean()));bc=float(torch.sqrt(((b-c[::2])**2).mean()))
    assert 0<bc<.8*ab
    assert min(r['minimum_J'] for r in rows)>.99
    assert max(r['force_residual_N'] for r in rows)<1.01e-10


def test_inactive_load_rejects_nonzero_action_without_dropping_it():
    X,tets,faces=tiny_geometry();patch,K,D=candidate_patches()
    loads=[boundary.GaussianPatch(p.center_material_m,p.covariance_material_m2,'known_hardware',active=(j==0)) for j,p in enumerate(patch)]
    m=NonlinearGaussianReference(X,tets,faces,young=5000,poisson=.35,density=625,
        support_patches=patch,stiffness=K,damping=D,load_patches=loads)
    assert torch.equal(m.load_active,torch.tensor([True,False]))
    before=m.load_weights.clone()
    # Even an arbitrarily small nonzero action must be rejected, not discarded.
    with pytest.raises(ValueError,match='Inactive load patch cannot carry force'):
        m.known_force([[.01,0,0],[0,1e-30,0]])
    assert torch.equal(m.load_weights,before)
    force=m.known_force([[.01,-.02,.03],[0,0,0]],gravity=[0,0,-.1])
    expected=tensor([.01,-.02,.03])+m.kernels.total_mass*tensor([0,0,-.1])
    torch.testing.assert_close(force.sum(0),expected,rtol=0,atol=1e-16)
    assert torch.count_nonzero(m.load_weights[1])==0
