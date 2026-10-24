import numpy as np
import pytest
from experiments import campaign_clamped_surface_nonlinear_v01 as s


def single():
    X=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    return s.NonlinearReference(X,np.array([[0,1,2,3]]),np.array([0]),(np.eye(4)+np.ones((4,4)))/20,young=4000.,support_K=10.)


def test_objective_rigid_rotation_and_rest():
    m=single();angle=.83
    Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    for x in (m.rest,m.rest@Q.T+[.03,-.01,.02]):
        energy,gradient,_,J=m.elastic(x,tangent=True)
        assert abs(energy)<1e-14
        assert np.linalg.norm(gradient)<1e-11
        np.testing.assert_allclose(J,1,atol=1e-14)


def test_energy_force_tangent_independent_finite_difference():
    m=single();rng=np.random.default_rng(244)
    u=rng.normal(size=m.rest.shape)*.001
    f=rng.normal(size=m.rest.shape)*.001
    v=rng.normal(size=m.rest.shape);v/=np.linalg.norm(v)
    e,r,H,J=m.evaluate(u,f,tangent=True)
    eps=1e-7
    ep,rp,_,_=m.evaluate(u+eps*v,f)
    em,rm,_,_=m.evaluate(u-eps*v,f)
    np.testing.assert_allclose((ep-em)/(2*eps),np.sum(r*v),rtol=2e-7,atol=2e-10)
    np.testing.assert_allclose((rp-rm).ravel()/(2*eps),H@v.ravel(),rtol=3e-8,atol=2e-7)
    np.testing.assert_allclose(H.toarray(),H.toarray().T,rtol=1e-13,atol=1e-12)


def test_zero_invalid_J_and_invalid_inputs():
    m=single();u,d=m.solve(np.zeros_like(m.rest))
    assert np.count_nonzero(u)==0 and d['iterations']==0
    for scale in (0.,-1.):
        x=m.rest.copy();x[:,0]*=scale
        with pytest.raises(ValueError):m.elastic(x)
    with pytest.raises(ValueError):m.solve(np.full_like(m.rest,np.nan))
    with pytest.raises(ValueError):m.solve(np.zeros_like(m.rest),initial_displacement=np.ones_like(m.rest))
    with pytest.raises(ValueError):s.NonlinearReference(m.rest,m.tets,m.fixed,np.eye(4),young=4e3,support_K=10)


def test_shared_rest_hessian_matches_linear_operator():
    from experiments.campaign_surface_traction_static_v01 import SparseStaticForward
    for n in (2,4):
        linear=SparseStaticForward(n);m=s.from_shared(n,young=4700.,support_K=17.)
        _,r,H,J=m.evaluate(np.zeros_like(m.rest),np.zeros_like(m.rest),tangent=True)
        target=4700*linear.Kunit+17*linear.W
        err=H-target
        assert np.max(np.abs(err.data),initial=0)<2e-10
        assert np.linalg.norm(r)==0
        assert m.rest.shape==linear.X.shape and np.array_equal(m.tets,linear.tets)


def test_shared_matched_load_equilibrium_reactions_and_small_load_limit():
    from scipy.sparse.linalg import spsolve
    from experiments.campaign_surface_traction_static_v01 import SparseStaticForward,surface_traction
    linear=SparseStaticForward(2);m=s.from_shared(2,young=4700.,support_K=17.)
    f,meta=surface_traction(2,np.array([0.,0.,-1.]),.128*1e-3)
    u,d=m.solve(f)
    H=4700*linear.Kunit+17*linear.W
    ul=np.zeros(m.rest.size);ul[m.free]=spsolve(H[m.free][:,m.free],f.ravel()[m.free])
    np.testing.assert_allclose(u.ravel(),ul,rtol=3e-3,atol=1e-10)
    assert d['free_residual_N']<=d['tolerance_N']
    assert d['force_balance_error_N']<2e-9
    assert d['moment_balance_error_Nm']<=d['moment_balance_tolerance_Nm']
    assert d['minimum_J']>.99 and np.count_nonzero(u[m.fixed])==0
    assert d['tangent_shift']==0.


def test_explicit_constitutive_energy_and_first_piola_oracle():
    m=single()
    F=np.array([[1.07,.08,-.03],[.02,.96,.04],[0.,-.02,1.03]])
    x=m.rest@F.T
    J=np.linalg.det(F);logJ=np.log(J)
    mu=m.young/(2*(1+m.poisson));lam=m.young*m.poisson/((1+m.poisson)*(1-2*m.poisson))
    exact_density=mu/2*(np.sum(F*F)-3)-mu*logJ+lam/2*logJ**2
    exact_P=mu*F+(lam*logJ-mu)*np.linalg.inv(F).T
    energy,gradient,_,actualJ=m.elastic(x,tangent=True)
    np.testing.assert_allclose(energy,m.volumes.sum()*exact_density,rtol=2e-13,atol=1e-15)
    exact_gradient=m.volumes[0]*np.einsum('iJ,aJ->ai',exact_P,m.gradients[0])
    np.testing.assert_allclose(gradient,exact_gradient,rtol=5e-14,atol=1e-13)
    np.testing.assert_allclose(actualJ,J,rtol=2e-15)
