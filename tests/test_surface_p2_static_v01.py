"""Prospective P2 operator controls; no production physical-case responses."""
import numpy as np
from experiments import campaign_surface_p2_static_v01 as s


def norm(A):return np.sqrt(np.sum(A.data**2))


def test_p2_counts_cut_patch_force_moment_and_signed_basis():
    for n,nodes,free,tets in[(1,81,189,24),(2,425,1125,192),(4,2673,7533,1536)]:
        X,T,faces,fixed,ids=s.geometry(n)
        assert X.shape==(nodes,3)and T.shape==(tets,10)and len(ids)==free and faces.shape[1]==6
        loads=s.qualification_loads(n)
        np.testing.assert_allclose(loads['normal'].sum(0),[0,0,-.1],atol=1e-12,rtol=0)
        np.testing.assert_allclose(loads['paired_axial'].sum(0),[.48,0,0],atol=1e-12,rtol=0)
        for force in loads.values():
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],force).sum(0),0,atol=1e-13,rtol=0)
            assert np.count_nonzero(force[fixed])==0
        _,meta=s.surface_traction(n,[0,0,-1],.1)
        assert abs(meta['patch_area_m2']-.0072)<1e-15 and abs(meta['integrated_weight_mass']-1)<1e-11
    _,meta=s.surface_traction(1,[0,0,-1],.1)
    assert np.any(meta['weights']<0) # signed consistent load coefficients mustnotbeclipped


def test_quadratic_interpolation_nested_piecewise_fields_and_query_matrix():
    rng=np.random.default_rng(2026102435);points=rng.uniform(size=(137,3))*[.36,.08,.08]
    coarse_X,*_=s.geometry(1);random=rng.normal(size=coarse_X.shape);expected=s.field_queries(1,random,points)
    polynomial=lambda X:np.column_stack([X[:,0]**2+X[:,1]*X[:,2],X[:,1]**2+X[:,0]*X[:,2],X[:,2]**2+X[:,0]*X[:,1]])
    for n in(1,2,4):
        X,*_=s.geometry(n);np.testing.assert_allclose(s.field_queries(n,polynomial(X),points),polynomial(points),atol=1e-15,rtol=1e-13)
        prolonged=s.field_queries(1,random,X);np.testing.assert_allclose(s.field_queries(n,prolonged,points),expected,atol=2e-13,rtol=2e-13)
    f=s.SparseStaticForward(1);u=rng.normal(size=f.X.shape);u[f.fixed]=0
    Q=s.point_query_matrix(f,points)
    np.testing.assert_allclose((Q@u.ravel()[f.free]).reshape(-1,3),s.field_queries(1,u,points),atol=1e-14,rtol=1e-13)


def test_positive_24_and80_bulk_operators_and_consistent_signed_surface_PSD():
    a=s.SparseStaticForward(1,quadrature_order=24);b=s.SparseStaticForward(1,quadrature_order=80)
    assert norm(a.Kunit-b.Kunit)/norm(b.Kunit)<1e-12
    W=a.W_scalar.toarray();assert np.any(W<0)
    np.testing.assert_allclose(W,W.T,atol=1e-16,rtol=0)
    assert np.linalg.eigvalsh(W).min()>-1e-13
    assert abs(W.sum()-1)<1e-10
    quad=s.support_quadrature_checks(1)
    assert quad['relative_16_vs_20']<1e-10
    assert quad['relative_12_vs_16']>1e-10 # preserved underresolved12diagnostic, not a waivedproductiongate
    assert quad['normalization_relative_error']<1e-10
    total=0.;second=np.zeros(3)
    for q,w in s.bulk_query_blocks(2,order=24):total+=w.sum();second+=w@(q*q)
    np.testing.assert_allclose(total,.002304,atol=1e-16)
    np.testing.assert_allclose(second/total,np.array([.36,.08,.08])**2/3,atol=1e-15)


def test_independent_existing_NH_rest_tangent_and_legacy_parent_solver():
    f=s.SparseStaticForward(1);E,K=3700.,11.;reference=s.P2NonlinearReference.from_forward(f,E,K)
    zero=np.zeros_like(f.X);state=reference.evaluate(zero,True)
    assert np.max(abs(state['residual']))<1e-14
    expected=E*f.Kunit+K*f.W
    assert norm(state['tangent']-expected)/norm(expected)<1e-12
    # Synthetic tinyloads only, notanyproductionfivecase response.
    force=s.qualification_loads(1)['normal']*1e-5
    linear=f.staticforward([[E,K]],force)[0,0]
    nonlinear,diag=reference.solve(force)
    np.testing.assert_allclose(nonlinear,linear,atol=2e-11,rtol=1e-4)
    assert diag['maximum_free_residual_N']<1e-10 and diag['search_J_guard']==.1
    assert not any(isinstance(value,np.ndarray)for value in diag.values())


def test_physical_gradient_and_strain_metrics_for_affine_field():
    f=s.SparseStaticForward(1);A=np.array([[.01,.002,0],[.002,-.003,.001],[0,.001,.004]])
    u=f.X@A.T;metrics=s.strain_metrics(f,u)
    np.testing.assert_allclose(metrics['minimum_J'],np.linalg.det(np.eye(3)+A),atol=1e-14)
    np.testing.assert_allclose(metrics['maximum_absolute_symmetric_strain_eigenvalue'],np.max(abs(np.linalg.eigvalsh(A))),atol=1e-14)
    assert metrics['checked_nodes_and24and80']and not metrics['whole_element_J_certified']


def test_refined_surface_quadrature_virtual_work_and_tiny_response_parity():
    f=s.SparseStaticForward(1);rng=np.random.default_rng(2026102436);u=rng.normal(size=f.X.shape)
    bary,qw=s.triangle_rule(20);N=s.triangle_shapes(bary);G=s.shared.gaussian_surface_normalization();direct=0.
    for ids in f.faces:
        xyz=f.X[ids[:3]];area=np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))/2;q=bary@xyz
        g=np.exp(-.5*np.sum((q-s.shared.SUPPORT_CENTER)**2,axis=1)/s.shared.SUPPORT_WIDTH**2)
        uq=N@u[ids];direct+=area*np.sum(qw*g*np.sum(uq*uq,axis=1))/G
    np.testing.assert_allclose(u.ravel()@(f.W@u.ravel()),direct,atol=1e-12,rtol=1e-11)
    for center in(.09,.135,.18,.225,.27):
        a,_=s.surface_traction(1,[0,0,-1],.1,center_x=center,order=16)
        b,_=s.surface_traction(1,[0,0,-1],.1,center_x=center,order=20)
        np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-10)
    force=s.qualification_loads(1)['normal']*1e-5;theta=[[3700.,11.]]
    f80=s.SparseStaticForward(1,quadrature_order=80)
    np.testing.assert_allclose(f.staticforward(theta,force),f80.staticforward(theta,force),atol=1e-14,rtol=1e-9)
    a,_=s.P2NonlinearReference.from_forward(f,3700.,11.,quadrature_order=24).solve(force)
    b,_=s.P2NonlinearReference.from_forward(f80,3700.,11.,quadrature_order=80).solve(force)
    np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-9)
