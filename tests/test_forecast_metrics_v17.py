import numpy as np
import pytest
from scripts.compositional_dynamics_v17.calibration_observations import queries,material_map
from scripts.compositional_dynamics_v17.forecast_metrics import reference_surface_weights,compare_surface_forecast


def test_rest_area_weights_and_translation_are_not_hidden():
    q,_=queries();rest=material_map(q);weights=reference_surface_weights(q,[.12,.06,.06])
    assert weights.sum()==pytest.approx(.036)
    x=np.broadcast_to(rest,(3,98,3)).copy();y=x+[.001,0,0];t=np.arange(3)*.01
    result=compare_surface_forecast(rest,q,[.12,.06,.06],t,x,t+1e-15,y)
    assert result['world_point_rms_m']==pytest.approx(.001)
    assert result['shape_point_rms_m']<1e-15
    assert result['surface_centroid_error_m']==pytest.approx(.001)
    assert 'com_error_m' not in result


def test_partial_predictions_count_missing_frames_and_time_shift_rejected():
    q,_=queries();rest=material_map(q);x=np.broadcast_to(rest,(3,98,3)).copy();t=np.arange(3)*.01
    result=compare_surface_forecast(rest,q,[.12,.06,.06],t,x,t[:2],x[:2])
    assert not result['complete'] and result['frame_coverage']==pytest.approx(2/3)
    with pytest.raises(ValueError):compare_surface_forecast(rest,q,[.12,.06,.06],t,x,t+.001,x)
