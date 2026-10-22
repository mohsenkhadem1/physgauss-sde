import numpy as np
from experiments.video_block_gravity_suite import align_to_rest,volume_weights
from experiments.boundary_block import block_mesh


def test_mass_weighted_rigid_removal_preserves_strain_and_rejects_reflection():
    X,T,_=block_mesh(size=(.24,.08,.08),cells=(4,2,2));w=volume_weights(X,T)
    angle=.6;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    q=X@R.T+[.3,-.2,.1];aligned,rms=align_to_rest(X,q,w)
    np.testing.assert_allclose(aligned,X,atol=2e-15);assert rms<1e-15
    deformed=X*np.array([1.04,1,1]);_,strain=align_to_rest(X,deformed@R.T+[.3,-.2,.1],w)
    assert strain>.001
    reflected=X*np.array([-1,1,1]);_,error=align_to_rest(X,reflected,w)
    assert error>.01
