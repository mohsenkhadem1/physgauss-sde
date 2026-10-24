import numpy as np
import pytest
import torch
from experiments import campaign_spatial_fixture_v01 as s

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2);yield;torch.set_num_threads(old)


def synthetic_units():
    return np.array([[.23,.31,.47,.59,.67,.71],[.72,.65,.42,.27,.34,.21],[.4,.1,.9,.2,.8,.5]])


def test_prior_explicit_six_dimensions_and_domain():
    t=s.physical(np.array([[0]*6,[1]*6],float))
    np.testing.assert_allclose(t[0],[2000,.005,.005,.005,.018,5])
    np.testing.assert_allclose(t[1],[8000,.095,.095,.095,.05,30])
    assert len(s.FUNCTION_NAMES)==20
    with pytest.raises(ValueError):s.physical(np.ones((1,4)))


def test_independent_fixture_assembly_and_30x30_solve():
    forward=s.SpatialForward();theta=s.physical(synthetic_units());K,C=forward.matrices(theta)
    forces=[s.old.known_force('A'),s.old.known_force('B'),list(s.loads.poke_programs().values())[5]]
    for j,th in enumerate(theta):
        ki,ci,m,w=s.independent_matrices(th)
        np.testing.assert_allclose(K[j],ki,rtol=2e-14,atol=2e-13)
        np.testing.assert_allclose(np.diag(C[j]),ci,rtol=2e-14,atol=1e-15)
        np.testing.assert_allclose(forward.weights(theta)[j],w,rtol=2e-14,atol=1e-15)
        for force in forces:
            np.testing.assert_allclose(forward(th[None],force)[0],s.independent_numpy_forward(th,force),rtol=2e-11,atol=1e-14)


def test_shared_eleven_rhs_features_match_scalar_oracle():
    forward=s.SpatialForward();theta=s.physical(synthetic_units()[:2]);target,pokes,functions=s.joint_features(forward,theta,chunk=1)
    assert target.shape==(2,135) and pokes.shape==(6,2,12) and functions.shape==(2,20)
    for j,th in enumerate(theta):
        reference=np.concatenate([s.independent_numpy_forward(th,force)[list(s.TARGET_FRAMES)].reshape(-1) for force in s.loads.force_programs().values()])
        np.testing.assert_allclose(target[j],reference,rtol=2e-11,atol=1e-14)
        for k,force in enumerate(s.loads.poke_programs().values()):
            ref=s.old.selected_observations(s.independent_numpy_forward(th,force)[None])[0]
            np.testing.assert_allclose(pokes[k,j],ref,rtol=2e-11,atol=1e-14)
        np.testing.assert_allclose(functions[j,:6],th)
        np.testing.assert_allclose(functions[j,6:11].sum(),th[5],rtol=1e-14)
        np.testing.assert_allclose(functions[j,-6:-3],s.independent_numpy_forward(th,s.old.known_force('A'))[-1,1],rtol=2e-11,atol=1e-14)
        np.testing.assert_allclose(functions[j,-3:],s.independent_numpy_forward(th,s.old.known_force('B'))[-1,4],rtol=2e-11,atol=1e-14)


def test_prefix_observation_only_and_generic_loglikelihood():
    forward=s.SpatialForward();theta=s.physical(synthetic_units());prefix=forward.prefix(theta)
    np.testing.assert_allclose(prefix,s.old.selected_observations(forward(theta,'A')),rtol=1e-13,atol=1e-15)
    observation=np.linspace(-1e-5,2e-5,12)
    expected=-.5*np.sum(((prefix-observation)/s.old.SIGMA)**2,axis=1)-12*np.log(s.old.SIGMA*np.sqrt(2*np.pi))
    np.testing.assert_allclose(forward.log_likelihood(synthetic_units(),observation),expected,rtol=1e-13)
    with pytest.raises(ValueError):forward.log_likelihood(synthetic_units(),np.zeros(13))


def test_six_dimensional_jacobian_refinement_and_independent_oracle():
    forward=s.SpatialForward();unit=synthetic_units()[0]
    J=s.observation_jacobian(forward,unit,step=1e-4);fine=s.observation_jacobian(forward,unit,step=5e-5)
    assert J.shape==(12,6) and np.all(np.linalg.norm(J,axis=0)>1e-7)
    np.testing.assert_allclose(J,fine,rtol=2e-5,atol=1e-10)
    oracle=[];h=1e-4
    for k in range(6):
        delta=np.eye(6)[k]*h
        pair=s.physical(np.array([unit+delta,unit-delta]))
        prediction=np.array([s.old.selected_observations(s.independent_numpy_forward(th,s.old.known_force('A'),steps=24)[None])[0] for th in pair])
        oracle.append((prediction[0]-prediction[1])/(2*h))
    np.testing.assert_allclose(J,np.array(oracle).T,rtol=2e-7,atol=1e-10)
    cov=np.diag(np.arange(1,7)*.001)
    score=s.local_fisher_scores(forward,unit,cov,step=1e-4)
    refined=s.local_fisher_scores(forward,unit,cov,step=5e-5)
    assert score.shape==(6,) and np.all(score>=0)
    np.testing.assert_allclose(score,refined,rtol=1e-5,atol=1e-10)


def test_exact_center_width_gauge_is_preserved_and_declared():
    forward=s.SpatialForward();a=s.physical(synthetic_units()[:1])[0]
    b=a.copy();factor=1.2;b[4]*=factor;b[1:4]=.05+factor**2*(a[1:4]-.05)
    assert np.all((b[1:4]>=.005)&(b[1:4]<=.095))
    assert a[4]!=b[4]
    np.testing.assert_allclose(forward.weights(a[None]),forward.weights(b[None]),rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(forward(a[None],'A'),forward(b[None],'A'),rtol=1e-12,atol=1e-14)
    assert not s.design()['parameter_recovery_claim']
    assert 'not separately identifiable' in s.design()['structural_nonidentifiability']
