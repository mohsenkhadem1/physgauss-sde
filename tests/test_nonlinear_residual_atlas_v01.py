import numpy as np
import pytest
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas,unit_axis,barycentric_basis


def polynomial(u):
    x,y=u.T
    return np.stack([x*x+2*y*y+x*y,1+3*x-y+x*x*y*y],axis=1)[:,None,:]*1e-6


def test_tensor_polynomial_reproduction_nodes_chunking_and_nesting():
    x=unit_axis(9);grid=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1)
    atlas=NonlinearResidualAtlas(polynomial(grid.reshape(-1,2)).reshape(9,9,1,2),['prefix'],provenance_fingerprint='synthetic')
    queries=np.random.default_rng(7401).uniform(size=(117,2))
    for n in(3,5,9):
        a=atlas.coarsen(n)
        np.testing.assert_allclose(a.predict(queries,['prefix'],chunk=7),polynomial(queries),rtol=1e-13,atol=1e-20)
        nodes=np.stack(np.meshgrid(a.axis,a.axis,indexing='ij'),axis=-1).reshape(-1,2)
        np.testing.assert_array_equal(a.predict(nodes,['prefix']),a.delta.reshape(-1,1,2))
        np.testing.assert_array_equal(unit_axis(9)[::(8//(n-1))],unit_axis(n))


def test_offgrid_error_detects_underresolved_correction():
    x=unit_axis(9);u=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1)
    def f(q):return (q[:,0]**4+q[:,1]**4)[:,None,None]*1e-5
    a=NonlinearResidualAtlas(f(u.reshape(-1,2)).reshape(9,9,1,1),['f'],provenance_fingerprint='synthetic')
    q=np.array([[.2,.7],[.4,.3]])
    assert np.max(abs(a.coarsen(3).predict(q,['f'])-f(q)))>1e-8
    np.testing.assert_allclose(a.coarsen(5).predict(q,['f']),f(q),atol=1e-20,rtol=1e-13)


def test_roundoff_halo_is_evaluated_without_parameter_clipping():
    x=unit_axis(3);u=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1)
    a=NonlinearResidualAtlas(u[:,:,0,None,None],['f'],provenance_fingerprint='synthetic')
    q=np.array([[-np.finfo(float).eps,.5]])
    assert a.predict(q,['f'])[0,0,0]<0
    with pytest.raises(ValueError):a.predict([[-1e-10,.5]],['f'])
    with pytest.raises(ValueError):a.predict([[.5,1.01]],['f'])
    with pytest.raises(ValueError):a.predict([[.5,.5]],['unknown'])
    with pytest.raises(ValueError):barycentric_basis([.5],[0,.4,1])
    with pytest.raises(ValueError):a.delta[0,0,0,0]=5
