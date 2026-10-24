import numpy as np
import pytest
import torch
from experiments import campaign_clamped_deformation_v02 as s

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2);yield;torch.set_num_threads(old)


def test_topology_fixed_design_netforce_and_C1_ramp():
    X,tets,faces,fixed,free=s.geometry()
    assert X.shape==(81,3) and tets.shape==(192,4) and len(fixed)==18 and len(free)==189
    vol=np.linalg.det((X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1))/6
    assert np.all(vol>0);np.testing.assert_allclose(vol.sum(),.36*.08*.08,rtol=1e-14)
    p=s.programs();np.testing.assert_allclose(p['bending'].sum(0),[0,0,.16]);np.testing.assert_allclose(p['stretching'].sum(0),[.72,0,0])
    for f in p.values():
        np.testing.assert_allclose(np.cross(X-[.18,.04,.04],f).sum(0),0,atol=1e-17)
    np.testing.assert_allclose(s.amplitude([0,.15,.3,.4]),[0,.5,1,1],atol=1e-15)
    epsilon=1e-6
    assert (1-float(s.amplitude(.3-epsilon)))/epsilon<3e-5
    assert s.GATES['minimum_nonrigid_RMS_m']==100e-6


def test_separate_numpy_assembly_matches_torch():
    forward=s.ExactClampedForward();K,m,w=s.independent_numpy_assembly()
    np.testing.assert_allclose(forward.Kunit,K,rtol=2e-13,atol=1e-15)
    np.testing.assert_allclose(forward.mass,m,rtol=2e-13,atol=1e-16)
    np.testing.assert_allclose(forward.diagonal,w,rtol=2e-13,atol=1e-16)


def test_exact_augmented_LTI_vs_independent_DOP853_lowload():
    forward=s.ExactClampedForward();theta=np.array([[4300.,13.]])
    force=np.stack(list(s.programs().values()))*1e-3
    exact=forward.exacttrajectory(theta,force,return_state=True)
    independent=s.independent_dop853(theta[0],force,return_state=True)
    assert exact['displacement'].shape==(1,2,41,81,3)
    assert np.count_nonzero(exact['displacement'][:,:,:,forward.fixed])==0
    assert np.count_nonzero(exact['velocity'][:,:,:,forward.fixed])==0
    assert exact['maximum_free_residual_N'].max()<1e-14
    np.testing.assert_allclose(exact['displacement'][0],independent['displacement'],rtol=1e-5,atol=1e-12)
    np.testing.assert_allclose(exact['velocity'][0],independent['velocity'],rtol=2e-4,atol=1e-11)
    # Exactlinearity andindependent RHS superposition; notaproductiongate.
    summed=forward.exacttrajectory(theta,force.sum(0))
    np.testing.assert_allclose(summed[:,0],exact['displacement'].sum(1),rtol=1e-8,atol=1e-14)


def test_gradient_analytical_and_invalid_times():
    forward=s.ExactClampedForward();A=np.diag([.01,-.003,.002]);u=(forward.X@A.T)[None]
    gradient=s.displacement_gradient(forward,u)
    np.testing.assert_allclose(gradient,np.broadcast_to(A,gradient.shape),atol=3e-17)
    with pytest.raises(ValueError):forward.exacttrajectory([[4000,10]],np.zeros((81,3)),times=np.arange(41)*.1)
