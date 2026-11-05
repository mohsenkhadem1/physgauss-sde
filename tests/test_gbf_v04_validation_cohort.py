import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import validation_cohort as cohort


def test_evaluation_requires_every_terminal_case(tmp_path):
    digest='a'*64
    for suite,task,episode in cohort.CASES:
        case=f'{suite}-task{task}-demo{episode}';path=tmp_path/case;path.mkdir()
        state={'case':case,'protocol_sha256':digest,'terminal':True,'stages':{s:{'state':'failed'}for s in cohort.STAGES}}
        (path/'prediction-status.json').write_text(json.dumps(state))
    assert len(cohort.terminal_gate(tmp_path,digest))==12
    state['stages']['physics']['state']='running';(path/'prediction-status.json').write_text(json.dumps(state))
    with pytest.raises(PermissionError):cohort.terminal_gate(tmp_path,digest)
    (path/'prediction-status.json').unlink()
    with pytest.raises(PermissionError):cohort.terminal_gate(tmp_path,digest)


def test_geometry_failure_preserves_baseline_and_blocks_all_physics(tmp_path,monkeypatch):
    calls=[]
    def command(args,*rest):calls.append(args);return 0
    monkeypatch.setattr(cohort,'launch',command)
    monkeypatch.setattr(cohort,'load_sealed',lambda path:None)
    def invalid(path):raise ValueError('disconnected convex64')
    monkeypatch.setattr(cohort,'preflight',invalid)
    args=SimpleNamespace(run_root=str(tmp_path),protocol='protocol',protocol_sha256='a'*64)
    state=cohort.execute_case(('object-soft',0,50),'cuda:0',args)
    assert state['terminal'] and state['stages']['baseline']['state']=='complete'
    assert state['stages']['geometry']['state']=='failed'
    assert all(state['stages'][s]['state']=='blocked'for s in ['physics','learned_contact','retrieval','retrieval_contact'])
    assert len(calls)==2 and not any('softvt_validation_score'in ' '.join(x)for x in calls)
    cohort.execute_case(('object-soft',0,50),'cuda:0',args)
    assert len(calls)==2 # no automatic retry


def test_preflight_requires_connected_positive_exact64(monkeypatch):
    import torch
    data={'nodes':np.zeros((63,3))};monkeypatch.setattr(cohort,'load_package',lambda *a:data)
    with pytest.raises(ValueError,match='exactly64'):cohort.preflight('unused')
    data['nodes']=np.zeros((64,3))
    model=SimpleNamespace(elements=torch.tensor([[0,1,2,3]]),mass=torch.ones(64))
    monkeypatch.setattr(cohort,'WideGaussianTableBoundaryBackend',lambda *a:SimpleNamespace(model=model))
    with pytest.raises(ValueError,match='disconnected'):cohort.preflight('unused')


def test_live_or_uncertain_status_cannot_become_terminal(tmp_path):
    case=('object-soft',0,50);folder=tmp_path/'object-soft-task0-demo50';folder.mkdir();path=folder/'prediction-status.json'
    state={'case':folder.name,'protocol_sha256':'a'*64,'terminal':False,'coordinator':cohort.process_identity(),
           'stages':{'physics':{'state':'running','child':cohort.process_identity(),'command':['test']}}}
    path.write_text(json.dumps(state));before=path.read_bytes()
    args=SimpleNamespace(run_root=str(tmp_path),protocol='protocol',protocol_sha256='a'*64)
    with pytest.raises(RuntimeError,match='Live or uncertain'):cohort.execute_case(case,'cuda:0',args)
    assert path.read_bytes()==before
    state['coordinator']['pid_namespace']='different-namespace';state['stages']['physics']['child']['pid_namespace']='different-namespace'
    path.write_text(json.dumps(state));before=path.read_bytes()
    with pytest.raises(RuntimeError):cohort.execute_case(case,'cuda:0',args)
    assert path.read_bytes()==before


def test_exclusive_lock_and_pid_reuse(tmp_path):
    with cohort.cohort_lock(tmp_path):
        with pytest.raises(RuntimeError,match='exclusive'): 
            with cohort.cohort_lock(tmp_path):pass
    identity=cohort.process_identity();assert cohort.process_state(identity)=='live'
    identity['start_ticks']='0';assert cohort.process_state(identity)=='gone'
    identity['pid_namespace']='different';assert cohort.process_state(identity)=='uncertain'


def test_launch_records_child_identity_before_wait(tmp_path):
    import sys
    captured=[];command=[sys.executable,'-c','import time;time.sleep(.1)']
    assert cohort.launch(command,tmp_path/'child.log',captured.append,3)==0
    assert captured[0]['command']==command and int(captured[0]['start_ticks'])>0
    assert cohort.process_state(captured[0])=='gone'
