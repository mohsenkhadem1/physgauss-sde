import json
import pytest
from experiments import audit_surface_p2_study_v03_failure_v01 as a


def test_controlled_stop_retains_nonzero_exits_but_requires_all_owned_closure(tmp_path,monkeypatch):
    monkeypatch.setattr(a.core,'require_absent',lambda pid:None)
    p=tmp_path/'execution.json';e=dict(protocol_SHA256='p',all_worker_groups_absent=True,pid=100,completed=False,status='FAILED',records=[dict(slot=i,pid=1000+i,reaped=True,exit_code=-15)for i in range(16)])
    p.write_text(json.dumps(e));assert len(a.closed_execution(tmp_path,'p')['records'])==16
    e['records'][3]['reaped']=False;p.write_text(json.dumps(e))
    with pytest.raises(ValueError,match='not reaped'):a.closed_execution(tmp_path,'p')
    e['records'][3]['reaped']=True;e['records'].pop();p.write_text(json.dumps(e))
    with pytest.raises(ValueError,match='All16'):a.closed_execution(tmp_path,'p')


def test_audit_source_has_no_target_response_or_displacement_array_access():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(a.__file__).read_text());keys=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript)and isinstance(node.value,ast.Name)and node.value.id=='z'and isinstance(node.slice,ast.Constant):keys.append(node.slice.value)
    assert set(keys)=={'unit','theta','force_SHA256','diagnostics_json'}
    assert a.DESIGN['no_policy_or_prediction_scores']
