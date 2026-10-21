import numpy as np
import pytest
import torch
from experiments import campaign_compliant_gaussian_linear as c


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(2)
    return c.manufactured(120)


def test_matrix_force_matches_independent_numpy_law_and_energy_gradient():
    X,tets,faces=c.tiny_geometry();q=c.boundary.SurfaceQuadrature.from_mesh(X,faces);patches,K,D=c.candidate_patches()
    adapter=c.GaussianFixtureMatrix.from_patches(q,patches,K,D)
    rng=np.random.default_rng(40);u=rng.normal(size=X.shape)*.001;v=rng.normal(size=X.shape)*.01
    target=rng.normal(size=(2,*X.shape))*.0003;tv=rng.normal(size=target.shape)*.002
    ref=c.boundary.compliant_support_response(q,patches,positions_world_m=X+u,velocities_world_m_s=v,
        target_positions_world_m=X[None]+target,target_velocities_world_m_s=tv,stiffness_world_N_m=K,damping_world_Ns_m=D)
    U=torch.tensor(u,dtype=torch.float64,requires_grad=True)
    values=[torch.tensor(a,dtype=torch.float64) for a in (v,target,tv)]
    got=adapter.response(U,*values);gradient=torch.autograd.grad(got['energy'],U)[0]
    np.testing.assert_allclose(got['forces'].detach().numpy(),ref['forces_world_N'],atol=3e-16,rtol=1e-12)
    assert float(got['energy'].detach())==pytest.approx(ref['elastic_energy_J'],abs=1e-17)
    assert float(got['dissipation'])==pytest.approx(ref['dissipation_rate_W'],abs=1e-17)
    torch.testing.assert_close(gradient,(adapter.K@U.flatten()-adapter.target_force(values[1],torch.zeros_like(values[2])).flatten()).reshape_as(U),atol=1e-16,rtol=1e-13)


def test_independent_discrete_manufacture_energy_mass_and_compatible_transport(case):
    row,_,_,_=case
    assert row['maximum_position_error_m']<2e-11 and row['maximum_velocity_error_m_s']<2e-9
    assert row['maximum_force_residual_N']<1e-9
    assert row['maximum_energy_balance_error_J']<1e-12
    assert row['maximum_balance_minus_residual_work_J']<1e-12
    assert row['total_physical_dissipation_J']>0 and row['total_numerical_dissipation_J']>0
    assert abs(row['total_target_work_J'])>1e-8
    assert row['minimum_J']>0 and row['maximum_F_error']<1e-10
    assert row['maximum_covariance_error_m2']<1e-13 and row['maximum_mass_conservation_error_kg']<1e-14
    assert row['all_DOFs_free'] and not row['hard_constraints_replaced']


def test_passive_observer_rotates_response_but_preserves_scalar_ledger(case):
    angle=.63;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    shifted,q,loads,_=c.manufactured(120,Q=Q,shift=np.array([.3,-.2,.7]))
    row,reference,forces,_=case
    np.testing.assert_allclose(q,reference@Q.T+[.3,-.2,.7],atol=2e-11,rtol=0)
    np.testing.assert_allclose(loads,forces@Q.T,atol=1e-10,rtol=1e-10)
    for key in ['total_external_work_J','total_target_work_J','total_physical_dissipation_J','total_numerical_dissipation_J','material_mass_kg']:
        assert shifted[key]==pytest.approx(row[key],rel=1e-9,abs=1e-13)


def test_invalid_negative_fixture_law_rejected_and_zero_modes_preserved():
    X,_,faces=c.tiny_geometry();q=c.boundary.SurfaceQuadrature.from_mesh(X,faces);patches,K,D=c.candidate_patches()
    bad=K.copy();bad[0,0,0]=-1.
    with pytest.raises(ValueError,match='PSD'):c.GaussianFixtureMatrix.from_patches(q,patches,bad,D)
    zero=np.zeros_like(K);law=c.GaussianFixtureMatrix.from_patches(q,patches,zero,zero)
    assert torch.count_nonzero(law.K)==0 and torch.count_nonzero(law.C)==0
    # The rank-one law is preserved, never promoted to SPD by repair.
    rank=c.GaussianFixtureMatrix.from_patches(q,patches,np.stack([K[1],K[1]]),np.stack([D[1],D[1]]))
    n=torch.tensor([1.,2.,-1.],dtype=torch.float64);null=torch.tensor([2.,-1.,0.],dtype=torch.float64)
    assert abs(float(n@null))==0
    assert torch.linalg.vector_norm(rank.K@null.repeat(len(X)))<1e-13
