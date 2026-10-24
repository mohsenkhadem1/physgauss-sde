import numpy as np
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference,point_query_matrix,posterior_grid,posterior_gauss,gauss_partial_CDF,compare_grids
from experiments.campaign_surface_traction_static_v01 import SparseStaticForward,qualification_loads,field_queries


def test_full_modal_identity_against_direct_sparse():
    f=SparseStaticForward(2);loads=np.stack(list(qualification_loads(2).values()));points=np.array([[.137,.021,.08],[.25,.04,.03]])
    P=point_query_matrix(f,points);modal=ModalStaticReference(f.Kunit_free,f.W_free,loads.reshape(2,-1).T[f.free],P)
    theta=np.array([[2000.,5.],[8000.,30.],[3500.,17.]])
    direct=f.staticforward(theta,loads);expected=field_queries(2,direct,points).transpose(0,2,3,1).reshape(3,6,2)
    np.testing.assert_allclose(modal(theta),expected,atol=2e-14,rtol=2e-10)
    assert modal.modes==len(f.free) and modal.diagnostics['all_modes_retained']


def test_uniform_normalization_continuous_cdf_and_moments():
    result=posterior_grid(lambda x:np.zeros(len(x)),n=33,predict=lambda x:x,keep=True)
    assert abs(result['log_evidence'])<1e-14
    np.testing.assert_allclose(result['unit_mean'],.5,atol=1e-14)
    np.testing.assert_allclose(result['unit_covariance'],np.eye(2)/12,atol=1e-14)
    np.testing.assert_allclose(result['unit_quantiles'],[[.05,.5,.95]]*2,atol=1e-14)
    np.testing.assert_allclose(result['predictive_SD'],np.sqrt(1/12),atol=1e-14)
    np.testing.assert_allclose(result['unit_CDF'],[np.linspace(0,1,33)]*2,atol=1e-14)


def test_truncated_gaussian_reference_converges_and_preserves_ridge():
    from scipy.special import ndtr
    mean=.37;sd=.08
    def ll(x):return -.5*((x[:,0]-mean)/sd)**2
    first=posterior_grid(ll,n=129,predict=lambda x:x*1e-3)
    second=posterior_grid(ll,n=257,predict=lambda x:x*1e-3)
    assert compare_grids(first,second)['passed']
    expected=(ndtr((np.linspace(0,1,33)-mean)/sd)-ndtr(-mean/sd))/(ndtr((1-mean)/sd)-ndtr(-mean/sd))
    np.testing.assert_allclose(second['unit_CDF'][0],expected,atol=1e-6)
    np.testing.assert_allclose(second['unit_quantiles'][1],[.05,.5,.95],atol=1e-14)


def test_independent_gauss_tensor_integral_against_closed_form_and_simpson():
    from scipy.special import ndtr
    mean=np.array([.33,.61]);sd=np.array([.065,.11])
    def ll(x):return -.5*np.sum(((x-mean)/sd)**2,axis=1)
    a=posterior_gauss(ll,n=64,predict=lambda x:x*1e-3)
    b=posterior_grid(ll,n=257,predict=lambda x:x*1e-3)
    Z=np.prod(sd*np.sqrt(2*np.pi)*(ndtr((1-mean)/sd)-ndtr(-mean/sd)))
    assert abs(a['log_evidence']-np.log(Z))<1e-12
    assert abs(a['log_evidence']-b['log_evidence'])<1e-7
    np.testing.assert_allclose(a['unit_mean'],b['unit_mean'],atol=1e-8)
    np.testing.assert_allclose(a['predictive_SD'],b['predictive_SD'],atol=1e-10)
    cdf=gauss_partial_CDF(ll,b['unit_quantiles'],n=64,log_evidence=a['log_evidence'])
    np.testing.assert_allclose(cdf,[[.05,.5,.95]]*2,atol=4e-5)


def test_partial_rectangle_CDF_uniform_and_boundary():
    ll=lambda x:np.zeros(len(x))
    q=np.array([[0.,.3,1.],[.05,.5,.95]])
    np.testing.assert_allclose(gauss_partial_CDF(ll,q,n=8,log_evidence=0),q,atol=1e-14)
