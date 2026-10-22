import numpy as np
from experiments.campaign_c38_bar_posterior import BarPosteriorBatch
from experiments.campaign_c38_bar_posterior_cached import CachedBarPosteriorBatch


def test_memoized_scalar_vector_parts_and_posteriors_are_bitwise_identical():
    for forces,sides,y in (([1.],[0],[[.00249],[.01001]]),([1.,2.],[1,0],[[.010,.007],[.013,.012]])):
        a=BarPosteriorBatch(y,forces,sides,conditional_order=32)
        b=CachedBarPosteriorBatch(y,forces,sides,conditional_order=32)
        for u in (0.,.1,.5,.7,1.,np.array([.2,.8])):
            for x,z in zip(a.parts(u),b.parts(u)):np.testing.assert_array_equal(x,z)
        np.testing.assert_array_equal(a.Z,b.Z)
        np.testing.assert_array_equal(a.weight(.3),b.weight(.3))
        for x,z in zip(a.moments(),b.moments()):np.testing.assert_array_equal(x,z)
        assert b._scalar_parts.cache_info().hits>0
