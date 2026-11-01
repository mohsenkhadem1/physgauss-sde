import numpy as np
import pytest
from scripts.benchmark_improvement_v14.phystwin_independent_audit import manual_score


def test_fixed_initial_correspondence_frame_mean_and_failure():
    initial=np.array([[0.,0,0],[1.,0,0]])
    pred=np.repeat(initial[None],3,axis=0);target=pred.copy()
    target[1,:,0]+=.002;target[2,:,0]+=.004
    assert manual_score(pred,target,1)==pytest.approx(3.)
    pred[2]=np.nan
    assert manual_score(pred,target,1)==pytest.approx(2.)
    pred[0]=np.nan
    assert manual_score(pred,target,1) is None
