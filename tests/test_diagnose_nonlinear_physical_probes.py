import numpy as np
import pytest
from experiments.diagnose_nonlinear_physical_probes import interpolate,rigid_registration,mass_properties
from experiments.campaign_nonlinear_excitation import geometry


def test_affine_interpolation_interior_surface_and_rigid_on_refined_meshes():
    X,T,F=geometry();P=np.vstack([X[T].mean(1),X[F].mean(1)])
    A=np.array([[1.04,.03,0],[0,.97,.02],[0,0,1.02]]);shift=np.array([.2,-.1,.3])
    for level in (0,1,2):
        points,tets,_=geometry(level);q=np.stack([points,points@A.T+shift])
        actual,_,_=interpolate(points,tets,q,P)
        np.testing.assert_allclose(actual[0],P,rtol=0,atol=2e-16)
        np.testing.assert_allclose(actual[1],P@A.T+shift,rtol=0,atol=2e-16)
    with pytest.raises(ValueError,match='outside'):interpolate(X,T,X[None],np.array([[1.,1.,1.]]))


def test_weighted_proper_rigid_registration_recovers_observer_transform():
    rng=np.random.default_rng(5);a=rng.normal(size=(2,12,3));w=np.arange(1,13)
    angle=.8;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    shift=np.array([2.,-1.,.3]);b=a@Q+shift
    found,translation=rigid_registration(a,b,w)
    np.testing.assert_allclose(np.einsum('tpi,tij->tpj',a,found)+translation[:,None],b,rtol=0,atol=2e-15)
    np.testing.assert_allclose(np.linalg.det(found),1,rtol=0,atol=2e-15)


def test_exact_uniform_tet_inertia_and_lumped_overestimate_refines():
    X=np.array([[0.,0,0],[1.,0,0],[0.,1.,0],[0.,0,1.]])
    props=mass_properties(X,np.array([[0,1,2,3]]),density=6.)
    expected=np.full((3,3),1/80);np.fill_diagonal(expected,3/40)
    np.testing.assert_allclose(props['exact_COM_m'],.25,rtol=0,atol=0)
    np.testing.assert_allclose(props['exact_inertia_COM_kg_m2'],expected,rtol=0,atol=2e-17)
    np.testing.assert_allclose(props['lumped_inertia_COM_kg_m2'],5*expected,rtol=0,atol=2e-16)
    results=[mass_properties(*geometry(i)[:2]) for i in range(3)]
    for r in results[1:]:np.testing.assert_allclose(r['exact_inertia_COM_kg_m2'],results[0]['exact_inertia_COM_kg_m2'],rtol=1e-14,atol=1e-18)
    errors=[r['relative_lumped_inertia_Frobenius_error'] for r in results]
    assert errors[2]<errors[1]<errors[0]
