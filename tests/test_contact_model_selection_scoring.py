import copy
import numpy as np
from scripts.external_benchmark.evaluate import score
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.contact_benchmark_v12.feasibility_verify import equal


def fixture():
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    pred=np.broadcast_to(points,(4,4,3)).copy()
    data=dict(dataset='PhysTwin',evaluation=dict(points=pred.copy(),mask=np.ones((4,4),bool),manual=pred.copy()))
    return data,pred


def test_finite_initial_is_exact_original_score():
    data,pred=fixture();pred[2]+=.01
    equal(safe_score(data,pred,1,4),score(data,pred,1,4))


def test_nonfinite_initial_leaves_valid_future_geometry_and_inputs_unchanged():
    data,pred=fixture();pred[0,0,0]=np.nan;before=pred.copy();original=copy.deepcopy(data)
    result=safe_score(data,pred,1,4)
    assert result['summary']['manual_track_mean_mm'] is None
    assert result['summary']['valid_frames']==3 and result['summary']['coordinate_rmse_mm']==0
    assert result['manual_correspondence_unavailable']['prediction_modified'] is False
    np.testing.assert_array_equal(pred,before);np.testing.assert_array_equal(data['evaluation']['points'],original['evaluation']['points'])
    assert data['dataset']=='PhysTwin'


def test_all_failed_prediction_retains_all_failed_frames():
    data,pred=fixture();pred[:]=np.nan
    result=safe_score(data,pred,1,4)
    assert result['summary']==dict(total_frames=3,valid_frames=0,failed_frames=3,manual_track_mean_mm=None)
    assert all(r['failed'] for r in result['frames'])
