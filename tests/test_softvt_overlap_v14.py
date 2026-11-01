import numpy as np
from scripts.benchmark_improvement_v14 import softvt_overlap as m


def test_budget_accounts_only_observed_slip(monkeypatch):
    n=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.1]])
    T,end=14,8;t=np.arange(T)*.1;o=np.repeat(n[None],end,axis=0);o[:,:,2]-=.04*t[:end,None]
    data=dict(dataset='SoftVTBench',nodes=n,train_end=end,observed=o,observed_mask=np.ones((end,4),bool),observed_initial=n,observation_matrix=np.eye(4),times=t,grasp=0,attachment_active=np.ones((T,2)),gripper_command=-np.ones(T))
    monkeypatch.setattr(m,'control_frames',lambda d:[(np.eye(3),np.zeros(3)) for _ in range(T)])
    monkeypatch.setattr(m,'rollout',lambda d,p,s:(p,{}))
    _,meta=m.predict(data,np.repeat(n[None],T,axis=0),dict(scene=None))
    assert np.isclose(meta['observed_spent_overlap_m'],.028)
    assert np.isclose(meta['remaining_overlap_m'],.022)
    assert meta['trigger']==13
