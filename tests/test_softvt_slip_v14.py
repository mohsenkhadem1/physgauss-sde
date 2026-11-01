import numpy as np
import pytest
from scripts.benchmark_improvement_v14 import softvt_slip as m


def test_prefix_only_trigger_and_rigid_shape(monkeypatch):
    nodes=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.02]],float)
    T,end=14,8;times=np.arange(T)*.1
    observed=np.repeat(nodes[None],end,axis=0);observed[:,:,2]-=times[:end,None]*.03
    d=dict(dataset='SoftVTBench',nodes=nodes,train_end=end,observed=observed,observed_mask=np.ones((end,4),bool),observed_initial=nodes,observation_matrix=np.eye(4),times=times,grasp=0,attachment_active=np.ones((T,2)),gripper_command=-np.ones(T))
    monkeypatch.setattr(m,'control_frames',lambda data:[(np.eye(3),np.zeros(3)) for _ in range(T)])
    monkeypatch.setattr(m,'rollout',lambda data,out,scene:(out,dict(command=data['gripper_command'].tolist())))
    base=np.repeat(nodes[None],T,axis=0)
    out,meta=m.predict(d,base,dict(scene=None))
    assert meta['trigger']==11
    assert meta['speed_m_s']==pytest.approx(.03)
    assert meta['contact']['command'][11]==1
    np.testing.assert_allclose(out[9]-out[9].mean(0),nodes-nodes.mean(0),atol=1e-14)
    # Prefix observations are exact permitted length; evaluator payload rejected.
    with pytest.raises(ValueError):m.predict(dict(d,evaluation={}),base,dict(scene=None))
    changed=base.copy();changed[12:]+=100
    _,other=m.predict(d,changed,dict(scene=None))
    assert other['trigger']==meta['trigger']
    assert other['slip_velocity_m_s']==meta['slip_velocity_m_s']
