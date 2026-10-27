import numpy as np
import pytest
from scripts.phystwin.consistent_fixture import P1Fixture,P2Fixture,query
from scripts.phystwin.active_support_reference import ActiveBlock


def test_exact_affine_boundary_energy_and_uniform_traction():
    for degree in [1,2]:
        for n in [2,4]:
            m=(P1Fixture if degree==1 else P2Fixture)(n,3000,800,'right_lower')
            u=np.zeros_like(m.rest);u[:,0]=.03*m.rest[:,1]
            energy=.5*np.sum(u*(m.fixture_scalar@u))
            assert np.isclose(energy,.5*800*.03**2*.08**2/12,rtol=1e-12)
            vertical=np.zeros_like(m.rest);vertical[:,0]=.03*(m.rest[:,2]+.02)
            assert np.isclose(.5*np.sum(vertical*(m.fixture_scalar@vertical)),.5*800*.03**2*.04**2/12,rtol=1e-12)
            assert np.allclose(m.left.sum(),1) and np.allclose(m.right.sum(),1)
    old=ActiveBlock(2,3000,800,'right_lower')
    u=np.zeros_like(old.rest);u[:,0]=.03*old.rest[:,1]
    assert .5*np.sum(u.ravel()*(old.spring_matrix@u.ravel()))>energy*1.1
    u[:,0]=.03*(old.rest[:,2]+.02)
    assert np.isclose(.5*np.sum(u.ravel()*(old.spring_matrix@u.ravel()))/(.5*800*.03**2*.04**2/12),3.)


def test_spatial_query_reproduces_affine_and_quadratic_fields():
    rng=np.random.default_rng(91);points=rng.uniform([0,-.04,-.04],[.24,.04,.04],(79,3))
    for cls in [P1Fixture,P2Fixture]:
        m=cls(2,3000,800,'left_lower');u=m.rest@np.diag([.01,.02,.03])+[.001,-.002,.003]
        assert np.allclose(query(m,u,points),points@np.diag([.01,.02,.03])+[.001,-.002,.003],atol=1e-14)
        assert np.allclose(query(m,u,points,True),np.diag([.01,.02,.03]),atol=1e-13)
    u=np.c_[m.rest[:,0]**2,m.rest[:,1]**2,m.rest[:,2]**2]
    assert np.allclose(query(m,u,points),points**2,atol=1e-14)


def test_consistent_fixture_directional_energy_and_tangent():
    m=P2Fixture(2,3000,800,'right_lower');m.set_action('held_oblique')
    rng=np.random.default_rng(9);u=rng.normal(size=m.rest.shape)*1e-5;d=rng.normal(size=u.shape);d/=np.linalg.norm(d)
    h=1e-6;s=m.evaluate(u,True);a=m.evaluate(u+h*d);b=m.evaluate(u-h*d)
    assert np.isclose((a['potential_J']-b['potential_J'])/(2*h),np.sum(s['residual']*d),rtol=2e-6,atol=1e-9)
    assert np.allclose((a['residual']-b['residual']).ravel()/(2*h),s['tangent']@d.ravel(),rtol=1e-5,atol=1e-6)


def test_gpu_cpu_log_material_assembly_agreement():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cpu=P2Fixture(2,3000,800,'right_lower');gpu=P2Fixture(2,3000,800,'right_lower',device='cuda:0')
    rng=np.random.default_rng(8);u=rng.normal(size=cpu.rest.shape)*1e-5
    a=cpu.evaluate(u,True);b=gpu.evaluate(u,True)
    assert np.isclose(a['potential_J'],b['potential_J'],rtol=1e-10,atol=1e-14)
    assert np.allclose(a['residual'],b['residual'],rtol=1e-10,atol=1e-12)
    delta=a['tangent']-b['tangent']
    assert np.max(abs(delta.data))<1e-9


def test_graded_geometry_and_sparse_ordering_preserve_problem():
    from experiments.campaign_static_p2_block import QuadraticBlock
    m=P2Fixture(2,3000,800,'right_lower',grading=True)
    assert np.isclose(m.volumes.sum(),.24*.08*.08)
    assert np.isclose(m.attachment_area,.08*.04)
    u=np.c_[m.rest[:,0]**2,m.rest[:,1]**2,m.rest[:,2]**2]
    rng=np.random.default_rng(81);p=rng.uniform([0,-.04,-.04],[.24,.04,.04],(55,3))
    assert np.allclose(query(m,u,p),p**2,atol=1e-14)
    m.set_action('held_oblique');a=m.solve(force_tolerance_N=1e-10);b=QuadraticBlock.solve(m,force_tolerance_N=1e-10)
    assert np.allclose(a['displacement_local_m'],b['displacement_local_m'],atol=1e-10,rtol=1e-8)


def test_parallel_direct_solver_matches_superlu():
    pytest.importorskip('pypardiso')
    m=P2Fixture(2,3000,800,'right_lower');m.set_action('held_oblique')
    a=m.solve();m.linear_solver='pardiso';b=m.solve()
    assert np.allclose(a['displacement_local_m'],b['displacement_local_m'],atol=1e-10,rtol=1e-8)
