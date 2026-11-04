import numpy as np
import pytest
from scipy.stats import norm

from scripts.gaussian_boundary_flow_v02.reference_posterior import (
    trapezoid_weights, weighted_quantile, summarize_grid, compare_quadratures,
)


def test_trapezoid_integrates_constant_and_linear_on_irregular_axis():
    x=np.array([-2.,-.4,0.,1.,3.]);w=trapezoid_weights(x)
    assert np.isclose(w.sum(),5.)
    assert np.isclose(w@x,(3.**2-(-2.)**2)/2)
    with pytest.raises(ValueError):trapezoid_weights([1.,0.])


def test_duplicate_weighted_quantiles_and_invalid_mass():
    a=weighted_quantile([0.,0.,1.,2.],[.1,.1,.6,.2],[.5])
    b=weighted_quantile([0.,1.,2.],[.2,.6,.2],[.5])
    assert np.allclose(a,b)
    with pytest.raises(ValueError):weighted_quantile([0.,1.],[0.,0.])


def test_correlated_gaussian_affine_quadrature_matches_analytic_posterior():
    center=np.array([.3,-.2]);S=np.array([[.04,.01],[.02,.08]])
    n=129;L=7.;axis=np.linspace(-L,L,n)
    w=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1)
    ll=-.5*(w*w).sum(-1)-np.log(2*np.pi)-np.log(abs(np.linalg.det(S)))
    r=summarize_grid(center,S,L,n,ll)
    cov=S@S.T;std=np.sqrt(np.diag(cov))
    assert abs(r['log_evidence'])<1e-9
    assert np.allclose(r['mean_q'],center,atol=1e-10)
    assert np.allclose(r['covariance_q'],cov,rtol=1e-8)
    expected=center[:,None]+std[:,None]*norm.ppf([.05,.5,.95])
    assert np.allclose(r['quantiles_q'],expected,atol=.001)
    assert r['outer_edge_mass']<1e-10


def test_convergence_gate_rejects_collapsed_or_shifting_estimate():
    p={'std_q':[.1,.2],'mean_q':[0.,0.],'log_evidence':0.,'quantiles_q':[[-.1,0,.1],[-.2,0,.2]]}
    c={**p,'mean_q':[.01,0.]}
    thresholds={'log_evidence_change':.001,'mean_change_in_posterior_sd':.005,'quantile_change_in_posterior_sd':.02}
    assert compare_quadratures(p,p,thresholds)['passed']
    assert not compare_quadratures(p,c,thresholds)['passed']
