"""Small operator/isolation tests; no test-object generation or scoring."""
import tempfile,time,json
from pathlib import Path
import numpy as np
import pytest
from scripts.physical_refinement_v15.mechanics_adapter import run,bounded,specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,CASES,sha

def test_budget_expiry():
 with pytest.raises(TimeoutError):
  with bounded(.02):time.sleep(.08)

def test_parameter_validation():
 with pytest.raises(ValueError):specification(dict(CASES['homogeneous'],material_gradient=np.nan),np.zeros((1,2,6)),2,3)
 with pytest.raises(ValueError):specification(CASES['homogeneous'],np.zeros((1,2,5)),2,3)
 with pytest.raises(ValueError):specification(dict(CASES['homogeneous'],layout='opposite_y'),np.zeros((1,2,6)),2,3)

def test_nonfinite_diagnostic_cannot_complete(monkeypatch):
 import scripts.physical_refinement_v15.mechanics_adapter as adapter
 original=adapter.SmoothMaterialFixture.sampled
 def invalid(model,state):
  sample=original(model,state);sample['residual_N']=float('nan');return sample
 monkeypatch.setattr(adapter.SmoothMaterialFixture,'sampled',invalid)
 root=BASE/'adapter-tests';root.mkdir(exist_ok=True);out=Path(tempfile.mkdtemp(dir=root))
 with pytest.raises(RuntimeError,match='Nonfinite'):run(CASES['homogeneous'],np.zeros((1,2,6)),out,n=2,device='cpu',budget_seconds=60)
 assert not (out/'complete.json').exists() and not (out/'pose-000/complete.json').exists()
 assert json.loads((out/'status.json').read_text())['state']=='failed'

def test_each_probe_and_resume_resets_to_initial(monkeypatch):
 import shutil
 import scripts.physical_refinement_v15.mechanics_adapter as adapter
 from scripts.position_ood.fixture import controls
 root=BASE/'adapter-tests';root.mkdir(exist_ok=True);out=Path(tempfile.mkdtemp(dir=root));poses=np.array([controls('stretch',.001),controls('translate_y',.006),controls('twist_x')]);captured=[];original=adapter.solve
 def traced(model,initial=None,**kwargs):
  captured.append(None if initial is None else initial.copy());return original(model,initial,**kwargs)
 monkeypatch.setattr(adapter,'solve',traced)
 run(CASES['homogeneous'],poses,out,n=2,device='cpu',budget_seconds=120)
 with np.load(out/'pose-000/private-state.npz') as saved:first=saved['nodal_displacement'].copy()
 assert captured[0] is None
 np.testing.assert_array_equal(captured[1],first);np.testing.assert_array_equal(captured[2],first)
 (out/'complete.json').unlink();shutil.rmtree(out/'pose-002');captured.clear()
 run(CASES['homogeneous'],poses,out,n=2,device='cpu',budget_seconds=120)
 assert len(captured)==1;np.testing.assert_array_equal(captured[0],first)

def test_zero_state_geometry_separation_cache_and_tamper():
 root=BASE/'adapter-tests';root.mkdir(exist_ok=True);out=Path(tempfile.mkdtemp(dir=root));poses=np.zeros((1,2,6))
 record=run(CASES['homogeneous'],poses,out,n=2,device='cpu',budget_seconds=60)
 before=sha(out/'complete.json');again=run(CASES['homogeneous'],poses,out,n=2,device='cpu',budget_seconds=60);assert record==again and sha(out/'complete.json')==before
 with np.load(out/'pose-000/private-state.npz') as x:np.testing.assert_array_equal(x['nodal_displacement'],np.zeros_like(x['nodal_displacement']))
 with np.load(out/'pose-000/reaction-wrench.npz') as x:
  np.testing.assert_array_equal(x['forces_N'],np.zeros((2,3)));np.testing.assert_array_equal(x['torques_Nm'],np.zeros((2,3)))
 with np.load(out/'pose-000/sensor-geometry.npz') as x:
  assert set(x.files)=={f'mesh{i}_{k}' for i in range(3) for k in ('vertices','triangles','color')}
 with pytest.raises(ValueError):run(dict(CASES['homogeneous'],E=24000),poses,out,n=2,device='cpu',budget_seconds=60)
 with open(out/'pose-000/surface.npz','ab') as f:f.write(b'tamper')
 with pytest.raises(ValueError):run(CASES['homogeneous'],poses,out,n=2,device='cpu',budget_seconds=60)
