import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import isaac_static_secondary as module


def test_static_prediction_keeps_node_identity_and_clock():
    rest=np.arange(455*3,dtype=np.float32).reshape(455,3)/1000
    pred=module.static_prediction(rest)
    assert pred.shape==(61,455,3)
    np.testing.assert_array_equal(pred,np.broadcast_to(rest,pred.shape))
    pred[0,0]=100
    assert not np.array_equal(pred[0],rest)
    np.testing.assert_array_equal(pred[1],rest)


def test_rigid_motion_removed_and_zero_ratio_guard():
    rng=np.random.default_rng(4);rest=rng.normal(size=(455,3));pred=np.repeat(rest[None],21,axis=0)
    target=pred+np.array([1,2,3])
    shape=module.primary.rigid_aligned_shape_rmse(pred,target)
    assert shape<1e-14
    assert module.amplitude_ratio(shape,2)<1e-14
    assert module.amplitude_ratio(0,0) is None
    with pytest.raises(ValueError):module.amplitude_ratio(1,float('nan'))


def test_entire_cohort_gate_precedes_target_access(tmp_path,monkeypatch):
    plan={'cases':[{'case':'missing'}]}
    monkeypatch.setattr(module,'verify',lambda output:(plan,{}))
    calls=[]
    monkeypatch.setattr(module.primary,'load_truth_after_gate',lambda *args:calls.append(args))
    with pytest.raises(FileNotFoundError):module.score(tmp_path)
    assert calls==[]
    assert not (tmp_path/'all-predictions-barrier.json').exists()


def test_true_shape_change_survives_rigid_alignment():
    rng=np.random.default_rng(9);rest=rng.normal(size=(455,3));pred=np.repeat(rest[None],21,axis=0)
    target=pred*np.array([1.5,1.,1.])
    total=float(np.sqrt(np.mean((pred-target)**2)))
    shape=module.primary.rigid_aligned_shape_rmse(pred,target)
    assert 0.9 < module.amplitude_ratio(shape,total) <= 1.000001
