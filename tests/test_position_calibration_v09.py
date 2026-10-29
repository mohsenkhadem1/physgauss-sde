import numpy as np
from scripts.position_ood.calibration import conformal_scale,standardized_score,coverage,FLOOR_M

def test_small_calibration_set_does_not_invent_finite_guarantee():
    assert np.isinf(conformal_scale([1.,2.,3.]))
    assert conformal_scale(np.arange(1.,13.))==12.

def test_object_score_matches_coordinate_coverage_target():
    truth=np.zeros((7,81,3));mean=np.arange(truth.size).reshape(truth.shape)*1e-7;std=np.ones_like(truth)*1e-4
    q=standardized_score(mean,std,truth)
    result=coverage(mean,std,truth,q)
    assert result['coordinate_coverage']>=.9
    assert result['coordinate_coverage']<.902
    assert np.isfinite(standardized_score(mean,np.zeros_like(std),truth))
