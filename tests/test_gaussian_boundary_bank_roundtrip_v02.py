import time
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.bank_roundtrip import metrics,deadline,DeadlineExceeded

def test_pose_shape_separation_and_reflection():
    p=np.random.default_rng(1).normal(size=(3,20,3));q=p+np.array([.01,0,0]);r=metrics(q,p)
    assert r['aligned_shape_rmse_mm']<1e-10 and r['centroid_rmse_mm']==pytest.approx(10/np.sqrt(3))
    assert metrics(-p,p)['aligned_shape_rmse_mm']>1

def test_deadline_fails_without_swallowing_and_restores():
    with pytest.raises(DeadlineExceeded):
        with deadline(.01):time.sleep(.02)
    with deadline(.1):pass
