import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.prediction_sanity import audit_arrays


def fixture():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    data=dict(name='test',nodes=nodes,observed_initial=nodes,observation_matrix=np.eye(3),
              observed=np.tile(nodes,(2,1,1)),observed_mask=np.ones((2,3),bool),train_end=2,times=np.arange(4),floor=0.)
    plan=dict(point_ids=[0,1,2],prefix_frames=[0,1],prediction_methods=['flow_raw'])
    raw=np.tile(nodes,(4,1,1))+[0,0,-.2]
    pred={k:data[k] for k in ['nodes','observed_initial','observation_matrix']};pred.update(flow_raw=raw,flow_raw_samples=np.stack([raw,raw]))
    complete=dict(forecast_samples={'flow':{'requested':3,'valid':2,'failures':[{'sample':2}]}})
    return data,plan,complete,pred


def test_constant_initial_offset_detected_despite_relative_cancellation():
    data,plan,complete,pred=fixture();r=audit_arrays(data,plan,complete,pred)['methods']['flow_raw']
    assert np.allclose(pred['flow_raw']-pred['flow_raw'][:1],0)
    assert r['mean_prediction']['initial_observation_coordinate_rmse_mm']['mean']==pytest.approx(200/np.sqrt(3))
    assert r['mean_prediction']['allframe_maximum_floor_penetration_mm']==pytest.approx(200)
    assert r['surviving_draws']['requested']==3 and r['stored_draws']==2


def test_known_correction_magnitude_and_mapping_validation():
    data,plan,complete,pred=fixture();corrected={k:pred[k] for k in ['nodes','observed_initial','observation_matrix']}
    corrected['flow_pose']=pred['flow_raw']+[0,0,.2]
    r=audit_arrays(data,plan,complete,pred,corrected)['methods']['flow_raw']['pose_contact_correction']
    assert r['coordinate_rms_mm']==pytest.approx(200/np.sqrt(3))
    assert r['maximum_node_translation_mm']==pytest.approx(200) and r['corrected_floor_penetration_mm']==0
    corrected['observation_matrix']=np.ones((3,3))
    with pytest.raises(ValueError,match='mapping'):audit_arrays(data,plan,complete,pred,corrected)


def test_no_future_target_or_unselected_observation_use():
    data,plan,complete,pred=fixture();first=audit_arrays(data,plan,complete,pred)
    data['observed'][1]+=999
    assert first==audit_arrays(data,plan,complete,pred)
    data['evaluation']={}
    with pytest.raises(ValueError,match='forbidden'):audit_arrays(data,plan,complete,pred)
    del data['evaluation'];data['observed']=np.tile(data['nodes'],(4,1,1))
    with pytest.raises(ValueError,match='Prefix-only'):audit_arrays(data,plan,complete,pred)


def test_nonfinite_prediction_remains_explicit():
    data,plan,complete,pred=fixture();pred['flow_raw'][3]=np.nan
    r=audit_arrays(data,plan,complete,pred)['methods']['flow_raw']
    assert not r['finite'] and 'mean_prediction' not in r
