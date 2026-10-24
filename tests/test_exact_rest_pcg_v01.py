"""Synthetic direction controls only; no new force-response qualification."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG,DirectionFailure


def test_SPD_direct_equivalence_multiple_tangents_one_rest_factorization():
    rng=np.random.default_rng(371);B=rng.normal(size=(40,40));rest=B.T@B+10*np.eye(40)
    solver=ExactRestPCG(sparse.csc_matrix(rest));LU=solver.rest_LU
    for amplitude in (.01,.1,.2):
        Q=rng.normal(size=(40,5));H=sparse.csc_matrix(rest+amplitude*Q@Q.T);b=rng.normal(size=40)*1e-3
        d,diag=solver.direction(H,b);direct=spsolve(H,b)
        assert diag['accepted']and not diag['fallback']
        assert np.linalg.norm(H@d-b)<=max(1e-13,1e-11*np.linalg.norm(b))
        assert np.max(abs(d-direct))<1e-13
    assert solver.rest_factorizations==1 and solver.rest_LU is LU
    assert np.allclose(solver.rest_solve(np.eye(40)[:,:2]),np.linalg.solve(rest,np.eye(40)[:,:2]),rtol=1e-13,atol=1e-14)
    assert solver.current_fallback_factorizations==0


def test_bounded_CG_failure_falls_back_to_original_LU():
    rest=sparse.eye(30,format='csc');H=sparse.diags(np.linspace(1,300,30),format='csc');b=np.ones(30)*1e-3
    solver=ExactRestPCG(rest,maximum_iterations=1);d,diag=solver.direction(H,b)
    assert diag['fallback']and diag['method']=='unmodified_current_LU_fallback'
    assert diag['CG_iterations']==1 and diag['accepted']
    assert np.allclose(d,spsolve(H,b),rtol=1e-14,atol=1e-16)
    assert solver.current_fallback_factorizations==1


def test_indefinite_negative_curvature_direction_fails_closed():
    solver=ExactRestPCG(sparse.eye(2,format='csc'));H=sparse.diags([-1.,2.],format='csc')
    with pytest.raises(DirectionFailure)as error:solver.direction(H,np.array([1e-3,0.]))
    diag=error.value.diagnostics
    assert diag['fallback']and not diag['accepted']and diag['RHS_dot_direction']<0
    assert not diag['current_tangent_modified']
    with pytest.raises(ValueError):solver.direction([[1,1],[0,1]],[1,1])
    zero,diag=solver.direction(sparse.eye(2),np.zeros(2))
    assert diag['method']=='zero_RHS'and np.array_equal(zero,np.zeros(2))


def test_P2_synthetic_non_equilibrium_tangent_direction_parity():
    from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,P2NonlinearReference
    fwd=SparseStaticForward(2);E,K=4000.,12.
    rest=E*fwd.Kunit_free+K*fwd.W_free;solver=ExactRestPCG(rest)
    # Prescribed tiny kinematic field, NOT a simulated force response or truth.
    u=np.zeros_like(fwd.X);u[:,2]=1e-5*np.sin(np.pi*fwd.X[:,0]/.36);u[fwd.fixed]=0
    nh=P2NonlinearReference.from_forward(fwd,E,K,quadrature_order=24)
    H=nh.evaluate(u,tangent=True)['tangent'][fwd.free][:,fwd.free]
    known=np.random.default_rng(739).normal(size=len(fwd.free))*1e-8;b=H@known
    d,diag=solver.direction(H,b);direct=spsolve(H,b)
    assert diag['accepted']and not diag['fallback']
    assert np.max(abs(d-known))<1e-15
    assert np.max(abs(d-direct))<1e-15
    assert diag['CG_iterations']<20
    assert solver.rest_factorizations==1


def test_nonfinite_CG_result_logged_as_unresolved_then_original_LU(monkeypatch):
    import json
    import experiments.campaign_exact_rest_pcg_v01 as helper
    monkeypatch.setattr(helper,'cg',lambda H,b,**kwargs:(np.full_like(b,np.nan),1))
    solver=ExactRestPCG(sparse.eye(3,format='csc'))
    d,diag=solver.direction(sparse.diags([1.,2.,3.],format='csc'),np.ones(3)*1e-3)
    assert diag['fallback']and diag['accepted']
    assert diag['CG_explicit_residual_N']is None and diag['CG_RHS_dot_direction']is None
    json.dumps(diag,allow_nan=False)
    assert np.allclose(d,[1e-3,.5e-3,1e-3/3])
