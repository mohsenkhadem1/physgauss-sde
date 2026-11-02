import pytest
from scripts.compositional_dynamics_v17.viscous_recovery import select_cases

def test_only_incomplete_and_missing_selected():
 a,b,c=({'name':n,'dt_s':.001}for n in ['a','b','c']);p=dict(cases=[a,b,c]);r=dict(terminal=True,rows=[dict(name='a',case=a,completed=True),dict(name='b',case=b,completed=False,error='TimeoutError')])
 assert select_cases(p,r)==[b,c]
 r['terminal']=False
 with pytest.raises(ValueError):select_cases(p,r)
def test_changed_case_rejected():
 with pytest.raises(ValueError):select_cases(dict(cases=[dict(name='a',dt=1)]),dict(terminal=True,rows=[dict(name='a',case=dict(name='a',dt=2),completed=False)]))

def test_changed_original_array_rejected(tmp_path):
 import json
 from scripts.compositional_dynamics_v17.viscous_recovery import validate_original_assets,sha
 folder=tmp_path/'original';folder.mkdir();f=folder/'trajectory.npz';f.write_bytes(b'original bytes')
 r=dict(path='original',plan_sha256='parent',completed=True,trajectory_sha256=sha(f));(folder/'complete.json').write_text(json.dumps(r));p=dict(recovery=dict(original_plan_sha256='parent'))
 validate_original_assets(p,dict(rows=[r]),tmp_path);f.write_bytes(b'changed')
 with pytest.raises(ValueError):validate_original_assets(p,dict(rows=[r]),tmp_path)
