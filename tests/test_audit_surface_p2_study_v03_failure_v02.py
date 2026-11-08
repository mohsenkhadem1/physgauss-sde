import json
import pytest
from experiments import audit_surface_p2_study_v03_failure_v02 as a


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


def test_actual_cpu12_child_guard_accepts_one_thread_and_rejects_two():
    import os,sys,subprocess
    script='from experiments import audit_surface_p2_study_v03_failure_v02 as a;a.guard()'
    env=dict(os.environ,PYTHONPATH=str(a.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    p=subprocess.run(['taskset','-c','12',sys.executable,'-c',script],cwd=a.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    p=subprocess.run(['taskset','-c','12',sys.executable,'-c',script],cwd=a.ROOT,env=dict(env,OMP_NUM_THREADS='2'),capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and 'One-thread audit guard'in p.stderr
    assert a.DESIGN['CPU']==12 and a.DESIGN['RSS_cap_bytes']==4*1024**3 and a.DESIGN['wall_cap_s']==900


def test_stopped_manifest_integrity_refuses_changed_completed_bytes(tmp_path,monkeypatch):
    path=tmp_path/'completed.json';path.write_text('{}');execution=tmp_path/'execution.json';execution.write_text('{}')
    m=tmp_path/'stopped-completed-manifest-LOCAL.json';m.write_text(json.dumps(dict(protocol_SHA256=a.PROTOCOL,execution_SHA256=a.core.sha(execution),all16_physics_retained=True,no_completed_bytes_modified=True,files={'completed.json':a.core.sha(path)})))
    monkeypatch.setattr(a,'MANIFEST_SHA',a.core.sha(m))
    o=tmp_path/'controlled-stop-ownership.json';o.write_text(json.dumps(dict(protocol_SHA256=a.PROTOCOL,controlled_stop=True,all16_workers_reaped=True,all17_PID_groups_absent=True,negative_run_preserved=True,completed_data_manifest_SHA256=a.MANIFEST_SHA,execution_SHA256=a.core.sha(execution))))
    monkeypatch.setattr(a,'STOP_SHA',a.core.sha(o))
    assert a.manifest_gate(tmp_path)==1
    path.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='completed artifact changed'):a.manifest_gate(tmp_path)
