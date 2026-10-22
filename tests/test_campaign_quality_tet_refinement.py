"""Pure geometry: conforming quality refinement, affine invariance, no solves."""
import numpy as np
from experiments import campaign_quality_tet_refinement as quality
from experiments.campaign_nonlinear_excitation import geometry as original_geometry


def test_domain_surface_conformity_original_vertices_and_positive_mass_alllevels():
    base,T0,F0=quality.geometry(0)
    for level in range(5):
        X,T,F=quality.geometry(level);m=quality.shape_metrics(X,T);c=quality.topology_certificate(X,T,base,F0)
        assert c['passed'] and len(T)==2*8**level and len(F)==6*4**level
        assert abs(m['volume_m3']-.0005)<1e-17
        np.testing.assert_array_equal(X[:5],base)
        assert m['minimum_cell_volume_m3']>0 and abs(625*m['volume_m3']-.3125)<1e-14


def test_quality_bounded_meanratio_radiusratio_and_hhalving_afterfirstlevel():
    meshes=[quality.geometry(i) for i in range(5)];metrics=[quality.shape_metrics(x,t) for x,t,_ in meshes]
    assert all(m['permutation_invariant_mean_ratio']['minimum']>.68 for m in metrics)
    assert all(m['radius_ratio_3rin_over_Rcircum']['minimum']>.62 for m in metrics)
    assert max(m['rest_Dm_singular_condition']['maximum'] for m in metrics)<5.83
    h=np.array([m['maximum_edge_length_m']['maximum'] for m in metrics])
    np.testing.assert_allclose(h[2:]/h[1:-1],.5,atol=1e-14)
    old=quality.shape_metrics(*original_geometry(4)[:2])
    assert old['rest_Dm_singular_condition']['maximum']>25 and h[-1]<old['maximum_edge_length_m']['maximum']/2


def test_exact_affine_compatible_F_and_energy_unchanged_alllevels():
    A=np.array([[1.06,.025,0.],[.01,.97,.015],[0.,0.,1.02]]);mu=5000/(2*1.35);lam=5000*.35/(1.35*.3);energies=[]
    for level in range(5):
        X,T,_=quality.geometry(level);q=X@A.T+[.02,-.01,.01]
        D=(X[T[:,1:]]-X[T[:,:1]]).transpose(0,2,1);d=(q[T[:,1:]]-q[T[:,:1]]).transpose(0,2,1)
        F=d@np.linalg.inv(D);np.testing.assert_allclose(F,np.broadcast_to(A,F.shape),atol=2e-14)
        J=np.linalg.det(F);W=.5*mu*(np.sum(F*F,axis=(1,2))-3)-mu*np.log(J)+.5*lam*np.log(J)**2
        energies.append(float(np.sum(np.linalg.det(D)/6*W)))
    np.testing.assert_allclose(energies,energies[0],atol=1e-14,rtol=0)


def test_geometry_ties_connectivity_invariant_passiverotation_translation_scale():
    X,T,_=quality.geometry(0);angle=.73;R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    Y=3.7*(X@R.T)+[.4,-.1,.2];S=T.copy()
    for _ in range(3):
        X,T,F,_=quality.refine_shortest(X,T);Y,S,G,_=quality.refine_shortest(Y,S)
        np.testing.assert_array_equal(T,S);np.testing.assert_array_equal(F,G)
        np.testing.assert_allclose(Y,3.7*(X@R.T)+[.4,-.1,.2],atol=4e-16)
