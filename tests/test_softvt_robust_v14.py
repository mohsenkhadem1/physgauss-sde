import numpy as np
from scripts.benchmark_improvement_v14 import softvt_robust as m

def test_finite_path_exact_and_nan_trigger(monkeypatch):
    base=np.arange(36.).reshape(3,4,3);expected=base+1
    monkeypatch.setattr(m,'frozen_predict',lambda d,b,c:(expected.copy(),{'unchanged':True}))
    monkeypatch.setattr(m,'rigid_nodal',lambda d:np.ones_like(base))
    monkeypatch.setattr(m,'rollout',lambda d,b,s:(b.copy(),{'scene':'same'}))
    data={'nodes':base[0].copy()}
    x,meta=m.predict(data,base,{'scene':None});assert np.array_equal(x,expected) and meta=={'unchanged':True}
    damaged=base.copy();damaged[2]=np.nan
    x,meta=m.predict(data,damaged,{'scene':None});assert np.isfinite(x).all() and meta['fallback_applied']
    assert meta['original_nonfinite_frames']==[2];assert np.array_equal(x[0],base[0])
    assert np.array_equal(x[1:],np.ones_like(x[1:]))

def test_failed_rigid_remains_explicit(monkeypatch):
    base=np.zeros((3,4,3));base[2]=np.nan
    monkeypatch.setattr(m,'frozen_predict',lambda d,b,c:(b.copy(),{}))
    monkeypatch.setattr(m,'rigid_nodal',lambda d:np.full_like(base,np.nan))
    monkeypatch.setattr(m,'rollout',lambda d,b,s:(b,{}))
    x,meta=m.predict({'nodes':base[0]},base,{'scene':None})
    assert np.array_equal(x,base,equal_nan=True) and meta['failure_retained'] and not meta['fallback_applied']
