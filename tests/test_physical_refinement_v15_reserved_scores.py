import json
import numpy as np
import pytest
from scripts.physical_refinement_v15.reserved_scores import committed,score_weights,block_scales,averaged


def test_commit_guard_rejects_missing_full_cohort(tmp_path):
    (tmp_path/'complete.json').write_text(json.dumps(dict(role='all_forecasts_committed_before_response_scoring',protocol_sha256='p',method_freeze_sha256='f',rows=[])))
    protocol=dict(reserved_worlds=[dict(name='dummy',initial_camera='left')],noise_repeats=1,tracks=['V'],fixed_sequences=[])
    with pytest.raises(ValueError,match='Missing declared forecast'):committed(tmp_path,protocol,'p','f')
    with pytest.raises(ValueError,match='Full frozen forecast commitment'):committed(tmp_path,protocol,'changed','f')


def test_score_exact_prediction_and_force_scale_mixture():
    fields=np.zeros((1,4,81,3));shapes=np.zeros((1,4,243));wrenches=np.ones((1,4,6))
    weights=np.array([.25,.75]);indices=np.array([0,0]);scales=np.array([1.,3.]);target=(fields[0],shapes[0],2.5*wrenches[0])
    result=score_weights(weights,fields,shapes,wrenches,indices,scales,target)
    for value in result['metrics'].values():assert abs(value)<1e-10
    assert result['highest_weight_index']==1
    assert result['highest_weight_metrics']['reaction_force_rmse_N']==.5
    assert result['proper_scores']['shape_m_crps_mean']==0
    # Exact two-atom CRPS: E|X-2.5| - .5E|X-X'| = .75-.375.
    assert result['proper_scores']['force_N_crps_mean']==pytest.approx(.375)


def test_engineering_blocks_receive_half_quarter_quarter_weight():
    scale=block_scales();x=np.r_[np.full(243,.001),np.full(3,.1),np.full(3,.01)]
    assert np.mean((x/scale)**2)==pytest.approx(1)
    assert np.sum((x[:243]/scale[:243])**2)/249==pytest.approx(.5)


def test_random_expectation_averages_errors():
    rows=[dict(metrics={'error':x},highest_weight_metrics={'error':x},proper_scores={'crps':x}) for x in (1.,9.)]
    assert averaged(rows)['metrics']['error']==5


def test_response_reader_rejects_wrong_pose_before_arrays(tmp_path,monkeypatch):
    from scripts.physical_refinement_v15 import reserved_scores as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    folder=tmp_path/'job';folder.mkdir();pose=folder/'pose-001';pose.mkdir()
    record={'signature':'s','pose_index':1,'pose_m_rad':np.zeros((2,6)).tolist()}
    (pose/'complete.json').write_text(json.dumps(record))
    (folder/'complete.json').write_text(json.dumps(dict(state='complete',signature='s',pose_completion_sha256={'1':module.sha(pose/'complete.json')})))
    (folder/'manifest.json').write_text(json.dumps(dict(signature='s',spec={'poses_m_rad':[np.zeros((2,6)).tolist()]*5})))
    with pytest.raises(ValueError,match='Response pose identity mismatch'):module.load_responses(folder,range(1,5))
