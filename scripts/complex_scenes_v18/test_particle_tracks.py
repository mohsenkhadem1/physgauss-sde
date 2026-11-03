import numpy as np
from scripts.complex_scenes_v18.particle_tracks import binding


def test_binding_reproduces_affine_surface_motion_including_extrapolation():
    counts=(12,6,6);q=np.random.default_rng(2).uniform(0,1,(30,3));q[:6]=[[0,0,0],[1,1,1],[0,.3,.4],[1,.2,.8],[.4,0,1],[.3,1,0]]
    particles=np.stack(np.meshgrid(*[(np.arange(n)+.5)/n for n in counts],indexing='ij'),-1).reshape(-1,3)
    A=np.array([[1.1,.2,.1],[.3,.9,.1],[.1,-.2,.8]]);offset=np.array([.1,.3,.5]);i,w=binding(q,counts)
    np.testing.assert_allclose(w.sum(1),1,atol=1e-7)
    np.testing.assert_allclose((w[...,None]*(particles@A.T+offset)[i]).sum(1),q@A.T+offset,atol=1e-7)
