import numpy as np
from scripts.complex_scenes_v18.score_fitted_development import metrics


def test_metrics_separate_rigid_motion_and_shape():
    rng=np.random.default_rng(4);actual=rng.normal(size=(3,2,20,3))*.02
    shifted=actual+np.array([.003,.004,0.]);result=metrics(actual,shifted)
    assert abs(result['vector_rms_mm']-5)<1e-10
    assert result['rigid_aligned_shape_rms_mm']<1e-10
    bent=actual.copy();bent[...,0]*=1.2
    assert metrics(actual,bent)['rigid_aligned_shape_rms_mm']>1
