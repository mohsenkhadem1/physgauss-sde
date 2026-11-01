import numpy as np
from scripts.physical_refinement_v15.supersampling_diagnostic import pixel_average
from scripts.physical_refinement_v15.feature_likelihood import features

def test_footprint_average_preserves_features_with_background():
    rng=np.random.default_rng(8)
    for factor in (2,4):
        rgb=rng.random((192*factor,256*factor,3))
        depth=rng.random((192*factor,256*factor))
        depth[::3,::5]=np.inf
        image=dict(rgb=rgb,depth=depth)
        low=pixel_average(image,factor)
        assert low['depth'].shape==(192,256)
        np.testing.assert_allclose(features(**image),features(**low),atol=1e-14)

def test_no_hit_depth_is_far_background_not_zero():
    image=dict(rgb=np.ones((8,8,3))*.25,depth=np.full((8,8),np.inf))
    result=pixel_average(image,2)
    np.testing.assert_array_equal(result['depth'],np.ones((4,4)))
    np.testing.assert_array_equal(result['rgb'],np.ones((4,4,3))*.25)
