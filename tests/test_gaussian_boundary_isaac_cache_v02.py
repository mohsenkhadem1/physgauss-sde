import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import isaac_prefix_cache as cache


def test_test_role_rejected_before_any_file_access(monkeypatch):
    def fail(*a,**k):raise AssertionError('public data must not be accessed')
    monkeypatch.setattr(cache,'load_public_observations',fail)
    with pytest.raises(ValueError,match='TRAIN/VALIDATION'):
        cache.cache_row(dict(path='/missing',recipe=dict(split='test')))


def test_wrong_prefix_clock_rejected_before_labels(monkeypatch):
    calls=[]
    def load(*a,**kw):
        calls.append(kw)
        return dict(cameras=dict(times_s=np.arange(10)*.25))
    monkeypatch.setattr(cache,'load_public_observations',load)
    with pytest.raises(ValueError,match='five declared'):
        cache.cache_row(dict(path='/missing',recipe=dict(split='train')))
    assert calls==[dict(size=(192,160),max_time_s=1.)]
