import numpy as np
import pytest
from scripts.benchmark_improvement_v14 import softvt_predict as m


def payload():
    d={k:None for k in m.REQUIRED}
    d.update(dataset='SoftVTBench',nodes=np.zeros((4,3)),train_end=2,observed=np.zeros((2,4,3)),observed_mask=np.ones((2,4),bool),times=np.arange(3))
    return d


def test_explicit_failures_preserve_original(monkeypatch):
    d=payload();b=np.arange(36.).reshape(3,4,3)
    monkeypatch.setattr(m,'simple_slip',lambda *a:(np.full_like(b,np.nan),{}))
    x,meta=m.predict(d,b,dict(scene=None));assert np.array_equal(x,b) and meta['fallback_to_original']
    with pytest.raises(ValueError):m.predict(dict(d,evaluation={}),b,dict(scene=None))
    with pytest.raises(ValueError):m.predict(d,b,dict(scene=None,window=9))
    b[2,0,0]=np.nan;x,meta=m.predict(d,b,dict(scene=None));assert np.array_equal(x,b,equal_nan=True)
