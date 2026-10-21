"""Manufactured inverse checks; no cohort/Isaac/GPU data or calls."""
import numpy as np
import pytest
import torch
from experiments import campaign_compliant_patch_inverse as inverse
from experiments.campaign_compliant_gaussian_linear import LinearCompliantSystem,GaussianFixtureMatrix,tiny_geometry
from experiments.campaign_boundary_gaussians import SurfaceQuadrature,GaussianPatch
from experiments.campaign_linear_fem import assemble_tetrahedra
from experiments.campaign_material_gaussians import MaterialGaussians

torch.set_num_threads(2)


def test_nonuniform_quadrature_integrates_normalized_physical_prior():
    theta,mass=inverse.prior_grid(5)
    assert theta.shape==(625,4) and np.all(mass>0) and np.ptp(mass)>0
    assert mass.sum()==pytest.approx(1.,abs=1e-14)
    assert mass@theta[:,0]==pytest.approx((8000.-2000.)/np.log(4),rel=1e-9)
    assert mass@theta[:,1]==pytest.approx(.035,abs=1e-15)
    assert np.min(np.abs(theta-inverse.TRUTH),axis=0).min()>1e-12


def test_constant_or_missing_observations_recover_prior_and_ambiguity():
    theta,mass=inverse.prior_grid(5)
    predicted=np.zeros((len(theta),3))
    constant,info=inverse.condition(predicted,np.zeros(3),mass)
    missing,empty=inverse.condition(predicted,np.full(3,np.nan),mass,observed_mask=np.zeros(3,bool))
    np.testing.assert_allclose(constant,mass,atol=1e-16)
    np.testing.assert_allclose(missing,mass,atol=1e-16)
    assert empty['observed_channels']==0
    assert inverse.parameter_summary(theta,missing)[1]['std']>0.01
    with pytest.raises(ValueError,match='normalized positive prior'):
        inverse.condition(predicted,np.zeros(3),mass*.5)


def test_informative_likelihood_contracts_exact_toy_predictive_distribution():
    candidate=np.array([[-1.],[0.],[1.]])
    mass=np.array([.2,.5,.3])
    weights,_=inverse.condition(candidate,np.array([0.]),mass,sigma=.1)
    prior_var=mass@(candidate[:,0]-mass@candidate[:,0])**2
    post_var=weights@(candidate[:,0]-weights@candidate[:,0])**2
    assert post_var<1e-10*prior_var and weights[1]>1-1e-12
    expected=mass*np.exp(-.5*candidate[:,0]**2/.1**2);expected/=expected.sum()
    np.testing.assert_allclose(weights,expected,atol=1e-15)


def test_batched_motion_matches_existing_bridge_with_same_fixed_known_load():
    theta=inverse.TRUTH[None]
    u,check=inverse.forward(theta,'A')
    X,tets,faces=tiny_geometry();E,center,width,k=theta[0]
    Q=SurfaceQuadrature.from_mesh(X,faces)
    patches=[GaussianPatch(np.array([center,.02,0.]),np.eye(3)*width**2,'candidate_boundary_hypothesis'),
        GaussianPatch(X[0],np.eye(3)*.03**2,'candidate_boundary_hypothesis',active=False)]
    fixture=GaussianFixtureMatrix.from_patches(Q,patches,[np.eye(3)*k,np.zeros((3,3))],
        [np.eye(3)*.05,np.zeros((3,3))])
    K,mass,_,_=assemble_tetrahedra(X,tets,E,.35,625.)
    model=LinearCompliantSystem(K,mass,fixture)
    state=torch.zeros((5,3),dtype=torch.float64);velocity=state.clone();target=torch.zeros((2,5,3),dtype=torch.float64)
    saved=[state.clone()];max_balance=0.
    for F in inverse.known_force('A'):
        state,velocity,ledger=model.step(state,velocity,target,target,torch.tensor(F),inverse.H)
        saved.append(state.clone());max_balance=max(max_balance,abs(ledger['balance_error_J']))
    np.testing.assert_allclose(u[0],torch.stack(saved).numpy(),atol=2e-15,rtol=1e-10)
    assert max_balance<1e-15 and check['maximum_physical_force_residual_N']<1e-11
    assert check['all_nodal_DOFs_free'] and check['hard_constraints']==0
    midpoint=(np.arange(60)+.5)/120
    active=(midpoint>.025)&(midpoint<.325)
    total=np.linalg.norm(inverse.known_force('A').sum(1),axis=1)
    assert total.max()<=.02 and np.all(total[~active]==0) and np.all(total[active]>0)


def test_material_frame_patch_passive_objectivity_and_compatible_shape_mass():
    theta=inverse.TRUTH[None];u,_=inverse.forward(theta,'A')
    angle=.52;Q=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    shifted,_=inverse.forward(theta,'A',Q=Q,shift=np.array([.4,-.2,.3]))
    np.testing.assert_allclose(shifted,u@Q.T,atol=2e-14,rtol=1e-10)
    X,tets,_=tiny_geometry();kernels=MaterialGaussians.from_tetrahedra(torch.tensor(X),torch.tensor(tets),625.)
    state=kernels.transport(torch.tensor(X+u))
    assert state.jacobians.min()>0
    torch.testing.assert_close(state.volumes*state.bulk_density,kernels.masses.expand_as(state.volumes),atol=1e-16,rtol=1e-14)
    assert torch.linalg.eigvalsh(state.covariances).min()>0


def test_each_unknown_affects_sparse_response_and_no_node_is_exactly_pinned():
    theta=np.repeat(inverse.TRUTH[None],5,axis=0)
    for j in range(4):theta[j+1,j]*=1.15
    u,_=inverse.forward(theta,'A')
    observed=inverse.selected_observations(u)
    assert np.all(np.max(np.abs(observed[1:]-observed[:1]),axis=1)>1e-9)
    assert np.min(np.max(np.abs(u[0]),axis=(0,2)))>1e-8
    assert np.all(inverse.matrices(theta)['weights'].numpy()>0)
