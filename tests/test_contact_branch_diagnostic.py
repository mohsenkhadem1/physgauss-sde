import copy
import numpy as np
import pytest
from scripts.contact_benchmark_v12.branch_diagnostic import continue_trajectory


def fixture():
    nodes=np.array([[x,y,z] for x in [0.,.04] for y in [0.,.04] for z in [.1,.14]])
    base=np.broadcast_to(nodes,(3,8,3)).copy()
    payload=dict(nodes=nodes,controller_targets=base[:,None].copy(),attachment_weights=np.ones((1,8)),attachment_active=np.ones((3,1)),floor=0.,times=np.arange(3)/30)
    theta=np.r_[[-4.,2.,5.],np.zeros(6),0.,0.]
    return payload,base,theta


def test_branch_diagnostic_uses_positions_only_and_does_not_mutate_them():
    payload,base,theta=fixture();before=copy.deepcopy(payload)
    prediction,meta=continue_trajectory(payload,base,theta,1,3,steps=2)
    assert not meta['failed'] and len(prediction)==2
    assert all(r['substeps'][-1]['fraction']==1 for r in meta['frames'])
    for key,value in before.items():np.testing.assert_array_equal(payload[key],value)
    payload['observed']=np.full((10,8,3),np.nan)
    changed,_=continue_trajectory(payload,base,theta,1,3,steps=2)
    np.testing.assert_array_equal(changed,prediction)


def test_branch_diagnostic_rejects_future_truth_and_invalid_interval():
    payload,base,theta=fixture()
    with pytest.raises(ValueError,match='Invalid bounded'):
        continue_trajectory(payload,base,theta,1,3,steps=0)
    payload['evaluation']={'points':base}
    with pytest.raises(ValueError,match='evaluator'):
        continue_trajectory(payload,base,theta,1,3)
