"""Object-level finite-sample ranks and simultaneous field-score semantics."""
import math
import numpy as np
import pytest
from experiments.campaign_simultaneous_bands import object_score, calibrate


def test_sixteen_objects_use_maximum_for_ninety_percent_band():
    result=calibrate(np.arange(1,17),alpha=.1)
    assert result.order_statistic_rank==16 and result.scale==16
    assert result.metadata()['calibration_objects']==16


def test_too_few_objects_for_requested_coverage_returns_infinite_band():
    result=calibrate([1.,2.,3.],alpha=.1)
    assert result.order_statistic_rank==4 and math.isinf(result.scale)
    assert result.metadata()['infinite_band'] and result.metadata()['scale'] is None


def test_finite_sample_rank_coverage_by_exact_leave_one_object_out():
    # Every possible held-out rank among17 exchangeable distinct scores.
    scores=np.arange(1.,18.)
    covered=[]
    for i,score in enumerate(scores):
        calibration=calibrate(np.delete(scores,i),alpha=.1)
        covered.append(score<=calibration.scale)
    assert sum(covered)==16  #16/17, including the finite-sample correction.


def test_complete_field_is_one_score_and_numerical_floor_is_explicit():
    mean=np.zeros((2,4,3))
    std=np.ones_like(mean)*.001
    truth=mean.copy();truth[1,3,2]=.003
    assert object_score(truth,mean,std)==3
    std[1,3,2]=0
    assert object_score(truth,mean,std)==pytest.approx(3e6)
    assert calibrate([.1,.2],alpha=.5).scale==1.  #Never shrink latent bands.


def test_invalid_shapes_and_negative_scales_are_not_broadcast_or_repaired():
    with pytest.raises(ValueError):object_score([1.,2.],[0.],[1.,1.])
    with pytest.raises(ValueError):object_score([1.],[0.],[-1.])
    with pytest.raises(ValueError):calibrate([1.,np.nan])


def test_reported_band_and_score_use_identical_floor_at_zero_variance():
    calibration=calibrate(np.full(16,2.),alpha=.1)
    low,high=calibration.bounds([0.],[0.])
    np.testing.assert_allclose([low[0],high[0]],[-2e-9,2e-9])
    assert calibration.contains([2e-9],[0.],[0.])
    assert not calibration.contains([2.01e-9],[0.],[0.])
