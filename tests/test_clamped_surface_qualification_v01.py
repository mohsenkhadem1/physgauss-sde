import numpy as np
import pytest
from experiments import run_clamped_surface_qualification_v01 as s


def test_proper_rigid_fit_and_nonrigid_coordinate_rms():
    rng=np.random.default_rng(12);X=rng.normal(size=(30,3));weights=rng.uniform(.5,1.5,len(X));a=.8
    Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    u=X@Q.T+np.array([1.,2,3])-X
    d=s.weighted_rigid_metrics(X,u,weights)
    assert d['nonrigid_RMS_m']<2e-15 and abs(d['proper_rotation_determinant']-1)<1e-14
    stretch=.1*X;d=s.weighted_rigid_metrics(X,stretch,weights)
    w=weights/weights.sum();center=w@X
    expected=np.sqrt(np.sum(w[:,None]*(.1*(X-center))**2)/3)
    np.testing.assert_allclose(d['nonrigid_RMS_m'],expected,rtol=2e-14)
    reflection=X.copy();reflection[:,0]*=-1
    assert s.weighted_rigid_metrics(X,reflection-X,weights)['nonrigid_RMS_m']>.1


def test_frozen_spatial_gate_and_failclosed_zero_order():
    d=s.spatial_gate(12e-6,3e-6)
    assert d['observed_order']==2 and d['spatial_passed'] and d['medium_adequate']
    assert not s.spatial_gate(4e-6,3e-6)['spatial_passed']
    assert not s.spatial_gate(4e-6,3e-6)['medium_adequate']
    assert not s.spatial_gate(50e-6,20e-6)['medium_adequate']
    assert not s.spatial_gate(0,0)['spatial_passed']
    with pytest.raises(ValueError):s.spatial_gate(np.nan,1.)


def test_affinity_environment_guard_rejects_before_execution(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('OMP_NUM_THREADS','1');monkeypatch.setenv('OPENBLAS_NUM_THREADS','1');monkeypatch.setenv('MKL_NUM_THREADS','1')
    monkeypatch.setattr(s.os,'sched_getaffinity',lambda _: {4})
    s.guard()
    monkeypatch.setenv('OMP_NUM_THREADS','2')
    with pytest.raises(ValueError):s.guard()
    monkeypatch.setenv('OMP_NUM_THREADS','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError):s.guard()


def test_common_weighted_rms_affine_fields_no_forward_solves():
    from experiments import campaign_surface_traction_static_v01 as f
    fields=[]
    for n in (2,4,8):
        X,*_=f.geometry(n);fields.append((n,X*.001))
    d=s.common_rms(fields)
    assert np.max(d)<1e-18
    X,*_=f.geometry(2);u=np.zeros_like(X);v=u.copy();v[:,0]=3e-5
    np.testing.assert_allclose(s.common_rms([(2,u),(2,v)])[1,0],3e-5/np.sqrt(3),rtol=1e-12)


def test_case_denominators_sources_and_distinct_gates():
    assert len(s.CASES)==5 and s.CASES[-1]['E_Pa']==4000.
    assert s.GATES['maximum_absolute_symmetric_strain_eigenvalue']==.05
    assert s.GATES['minimum_J']==.8 and s.nonlinear.GATES['minimum_J']==.1
    pins=s.sources()
    for p in ('experiments/run_clamped_surface_qualification_v01.py','experiments/campaign_surface_traction_static_v01.py','experiments/campaign_clamped_surface_nonlinear_v01.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/campaign_p2_keast_quadrature.py'):
        assert p in pins


def test_stability_diagnostic_detects_antisymmetric_negative_mode():
    from scipy import sparse
    good=sparse.diags([1.,2.,3.,4.],format='csr')
    assert s.stability_diagnostic(good,2)['passed']
    bad=np.diag([1.,1.,4.,5.]);bad[0,1]=bad[1,0]=2.
    d=s.stability_diagnostic(sparse.csr_matrix(bad),2)
    assert not d['passed'] and d['minimum_eigenvalue_estimate_N_m']<-.99
