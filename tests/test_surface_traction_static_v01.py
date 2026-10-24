"""Operator, quadrature, physical load and interpolation controls only.

No production five-case responses, posterior draws or outcome-selected controls.
"""
import numpy as np
import torch
from scipy import sparse
from experiments import campaign_surface_traction_static_v01 as s
from experiments.campaign_linear_fem import assemble_tetrahedra


def frob(A):return np.sqrt(np.sum(A.data**2))


def test_nested_positive_geometry_and_matched_surface_force_moment():
    for n in(2,4,8):
        X,tets,faces,fixed,free=s.geometry(n)
        assert len(X)==(4*n+1)*(n+1)**2 and len(tets)==24*n**3
        volume=np.linalg.det((X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1))/6
        assert np.all(volume>0);np.testing.assert_allclose(volume.sum(),.002304,rtol=2e-14)
        assert len(fixed)==2*(n+1)**2 and len(free)==3*(len(X)-len(fixed))
        loads=s.qualification_loads(n)
        np.testing.assert_allclose(loads['normal'].sum(0),[0,0,-.10],atol=1e-12,rtol=0)
        np.testing.assert_allclose(loads['paired_axial'].sum(0),[.48,0,0],atol=1e-12,rtol=0)
        for f in loads.values():
            assert np.count_nonzero(f[fixed])==0
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],f).sum(0),0,atol=1e-13,rtol=0)
        _,meta=s.surface_traction(n,[0,0,-1],.1)
        np.testing.assert_allclose(meta['patch_area_m2'],.0072,rtol=2e-14)
        assert abs(meta['integrated_weight_mass']-1)<1e-11


def test_support_consistent_quadrature_exact_common_normalization():
    X,_,faces,_,_=s.geometry(2);matrices={};diagnostics={}
    for order in(4,8,12,16):matrices[order],diagnostics[order]=s.consistent_support(X,faces,order=order)
    W=matrices[12]
    assert sparse.isspmatrix_csr(W) and np.count_nonzero(W.toarray()-np.diag(W.diagonal()))>0
    np.testing.assert_allclose(W.toarray(),W.T.toarray(),atol=1e-16,rtol=0)
    assert np.linalg.eigvalsh(W.toarray()).min()>-1e-14
    assert diagnostics[12]['normalization_relative_error']<=s.GATES['support_normalization_relative_error']
    assert frob(matrices[12]-matrices[16])/frob(matrices[16])<=s.GATES['support_12_vs_16_relative_Frobenius']
    assert frob(matrices[8]-matrices[12])/frob(matrices[12])<=s.GATES['support_8_vs_12_relative_Frobenius']
    # Coarse order4 remains a disclosed diagnostic, never normalized to force agreement.
    assert diagnostics[4]['exact_normalization_m2']==diagnostics[16]['exact_normalization_m2']
    assert not np.array_equal(matrices[4].data,matrices[12].data)
    for n in(4,8):
        xx,_,ff,_,_=s.geometry(n);_,d=s.consistent_support(xx,ff)
        assert d['normalization_relative_error']<=s.GATES['support_normalization_relative_error']


def test_traction_quadrature_orders_and_surface_work_identity():
    X,_,_,_,_=s.geometry(2);A=np.array([[.01,.002,0],[.003,-.004,.001],[0,.002,.005]]);u=X@A.T+[.001,.002,.003]
    force={k:s.qualification_loads(2,order=k)for k in(4,8,12,16)}
    for name in('normal','paired_axial'):
        np.testing.assert_allclose(force[12][name],force[16][name],atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(force[8][name],force[12][name],atol=1e-12,rtol=1e-8)
        assert abs(np.sum((force[12][name]-force[16][name])*u))<1e-14


def test_sparse_bulk_assembly_independent_torch_patch_and_rigid_controls():
    torch.set_num_threads(2);f=s.SparseStaticForward(2);K,m,g,v=assemble_tetrahedra(f.X,f.tets,1.,.35,625.)
    np.testing.assert_allclose(f.Kunit.toarray(),K.numpy(),atol=2e-15,rtol=1e-12)
    np.testing.assert_allclose(f.mass,m.numpy(),atol=1e-16,rtol=1e-13)
    np.testing.assert_allclose(f.mass[::3].sum(),1.44,rtol=1e-14)
    np.testing.assert_allclose(f.gradients,g.numpy(),atol=1e-13,rtol=1e-13)
    A=np.array([[.01,.002,.003],[.002,-.004,.001],[.003,.001,.005]])
    u=(f.X@A.T).reshape(-1);mu=1/(2*1.35);lam=.35/(1.35*.3)
    expected=(.5*lam*np.trace(A)**2+mu*np.sum(A*A))*.002304
    np.testing.assert_allclose(.5*u@(f.Kunit@u),expected,atol=1e-18,rtol=1e-11)
    omega=np.array([[0,-.03,.02],[.03,0,-.01],[-.02,.01,0]])
    rigid=(f.X@omega.T+[.001,.002,.003]).reshape(-1)
    assert np.max(abs(f.Kunit@rigid))<1e-16


def test_static_tiny_load_sparse_vs_independent_dense_no_production_response():
    f=s.SparseStaticForward(2);theta=np.array([[3700.,11.]])
    loads=np.stack(list(s.qualification_loads(2).values()))*1e-4
    actual,d=f.staticforward(theta,loads,return_diagnostics=True)
    assert actual.shape==(1,2,len(f.X),3) and np.count_nonzero(actual[...,f.fixed,:])==0
    K,_,_,_=assemble_tetrahedra(f.X,f.tets,3700.,.35,625.)
    dense=K.numpy()+11*f.W.toarray();rhs=loads.reshape(2,-1)[:,f.free].T
    expected=np.linalg.solve(dense[np.ix_(f.free,f.free)],rhs).T
    np.testing.assert_allclose(actual[0].reshape(2,-1)[:,f.free],expected,atol=2e-14,rtol=1e-9)
    assert d[0]['maximum_free_residual_N']<=s.GATES['free_residual_N']


def test_affine_physical_queries_and_common_positive_bulk_quadrature():
    rng=np.random.default_rng(2026102433);points=rng.uniform(size=(63,3))*[.36,.08,.08];points=np.concatenate([points,[[0,0,0],[.36,.08,.08],[.135,.04,.08],[.225,.04,.08]]])
    A=np.array([[.01,.003,.002],[.001,-.004,.002],[.003,.002,.005]]);translation=[.001,.002,.003]
    for n in(2,4,8):
        X,*_=s.geometry(n);field=X@A.T+translation
        np.testing.assert_allclose(s.field_queries(n,field,points),points@A.T+translation,atol=3e-17,rtol=0)
    total=0.;centroid=np.zeros(3)
    for q,w in s.bulk_query_blocks(4,tet_chunk=37):
        assert np.all(w>0);total+=w.sum();centroid+=w@q
    np.testing.assert_allclose(total,.002304,atol=1e-16)
    np.testing.assert_allclose(centroid/total,[.18,.04,.04],atol=1e-15)


def test_nested_piecewise_affine_prolongation_not_only_global_affine():
    rng=np.random.default_rng(2026102434);coarse_X,*_=s.geometry(2);coarse=rng.normal(size=coarse_X.shape)
    points=rng.uniform(size=(521,3))*[.36,.08,.08];expected=s.field_queries(2,coarse,points)
    for n in(4,8):
        fine_X,*_=s.geometry(n);prolongated=s.field_queries(2,coarse,fine_X)
        np.testing.assert_allclose(s.field_queries(n,prolongated,points),expected,atol=2e-14,rtol=2e-14)
    total=0.;second=np.zeros(3)
    for q,w in s.bulk_query_blocks(4,tet_chunk=37):total+=w.sum();second+=w@(q*q)
    np.testing.assert_allclose(second/total,np.array([.36,.08,.08])**2/3,atol=2e-16,rtol=2e-14)
