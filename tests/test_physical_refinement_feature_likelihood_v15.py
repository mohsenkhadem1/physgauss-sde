import numpy as np
from scripts.physical_refinement_v15.feature_likelihood import features,covariance,compare

def test_same_images_zero_information_and_covariance_positive():
    rgb=np.full((192,256,3),.5);depth=np.full((192,256),.3);a=dict(rgb=rgb,depth=depth)
    C=covariance(rgb,depth,.0003,.005)
    assert np.linalg.eigvalsh(C).min()>0
    for row in compare(a,a)['noise_sweep']:
        assert row['mahalanobis_squared']==0 and row['gaussian_KL_nats']==0
        assert row['equal_prior_gaussian_classification_error']==.5

def test_correlated_bias_prevents_pixel_count_as_independent_replicates():
    rgb=np.full((192,256,3),.5);depth=np.full((192,256),.3)
    a=dict(rgb=rgb,depth=depth);b=dict(rgb=rgb,depth=depth+.0003)
    x=compare(a,b)['noise_sweep'][1]
    assert .1<x['depth_only_mahalanobis_squared']<2
    assert x['depth_only_mahalanobis_squared']<192*256/100
