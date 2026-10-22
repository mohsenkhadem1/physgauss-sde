"""Positive quadrature/Galerkin identities, no mechanics integrations."""
import numpy as np
from experiments.campaign_consistent_surface_gaussians import triangle_rule,assemble_surface_gaussians
from experiments.campaign_quality_tet_refinement import geometry


def test_duffy_rule_constant_affine_and_quadratic_triangle_integrals():
    vertices=np.array([[.1,.2,-.3],[.4,.1,.2],[-.2,.6,.3]]);b,w=triangle_rule(5);x=b@vertices
    jac=np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]));area=jac/2;w=w*jac
    assert np.all(w>0) and abs(w.sum()-area)<1e-15
    np.testing.assert_allclose(w@x,area*vertices.mean(0),atol=1e-16)
    second=area/12*(vertices.T@vertices+np.outer(vertices.sum(0),vertices.sum(0)))
    np.testing.assert_allclose(np.einsum('q,qi,qj->ij',w,x,x),second,atol=1e-16)


def bank():return np.array([[.01,.02,.01],[.095,.09,.085]]),np.broadcast_to(np.diag(np.array([.035,.045,.03])**2),(2,3,3)).copy()


def test_normalization_rowsum_PSD_and_exact_affine_fixture_field_moments():
    X,T,F=geometry(1);C,S=bank();a=assemble_surface_gaussians(X,F,C,S,order=16)
    np.testing.assert_allclose(a.load_vectors.sum(1),1.,atol=3e-16)
    A=np.array([[.1,.2,.3],[-.2,.1,.05],[.3,-.1,.07]]);shift=np.array([.02,-.01,.03]);K=np.array([[3.,.2,0.],[.2,2.,.1],[0.,.1,1.]])
    u=X@A.T+shift
    for j,M in enumerate(a.fixture_grams):
        np.testing.assert_allclose(M@np.ones(len(X)),a.load_vectors[j],atol=1e-16)
        np.testing.assert_allclose(M.toarray(),M.toarray().T,atol=3e-17)
        assert np.linalg.eigvalsh(M.toarray()).min()>-1e-16
        actual=.5*np.sum(u*(M@u@K.T))
        Q=A.T@K@A;linear=A.T@K@shift
        expected=.5*np.sum(Q*a.second_moments_material_m2[j])+linear@a.centroids_material_m[j]+.5*shift@K@shift
        np.testing.assert_allclose(actual,expected,atol=2e-18)


def test_refined_mesh_affine_field_energy_and_moments_converge_to_same_surface_integral():
    C,S=bank();operators=[]
    for level in [0,1,2]:
        X,T,F=geometry(level);operators.append(assemble_surface_gaussians(X,F,C,S,order=20))
    for a in operators[1:]:
        np.testing.assert_allclose(a.normalizers_m2,operators[0].normalizers_m2,rtol=1e-12,atol=1e-16)
        np.testing.assert_allclose(a.centroids_material_m,operators[0].centroids_material_m,rtol=0,atol=3e-15)
        np.testing.assert_allclose(a.second_moments_material_m2,operators[0].second_moments_material_m2,rtol=0,atol=5e-16)


def test_passive_rotation_translation_preserves_scalar_load_and_fixture_operators():
    X,T,F=geometry(1);C,S=bank();a=assemble_surface_gaussians(X,F,C,S,order=12)
    angle=.4;R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]]);shift=np.array([.3,-.2,.1])
    b=assemble_surface_gaussians(X@R.T+shift,F,C@R.T+shift,np.einsum('ij,sjk,lk->sil',R,S,R),order=12)
    np.testing.assert_allclose(a.load_vectors,b.load_vectors,atol=2e-16)
    for A,B in zip(a.fixture_grams,b.fixture_grams):np.testing.assert_allclose(A.toarray(),B.toarray(),atol=2e-16)
