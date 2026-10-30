"""Independent orchestration tests; synthetic arrays only, no evaluator access."""
import json
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.prefix_ridge_switch_v13 import batch,score

def setup_paths(tmp_path,monkeypatch):
    base=tmp_path/'new';parent=tmp_path/'old';case={'name':'toy','cohort':'development','spec':{'case':'toy'}}
    monkeypatch.setattr(batch,'BASE',base);monkeypatch.setattr(batch,'PARENT',parent);monkeypatch.setattr(batch,'ROOT',tmp_path)
    monkeypatch.setattr(batch,'check_inputs',lambda paths:None)
    directory=parent/'model-selection-evaluation/development/toy';directory.mkdir(parents=True)
    old=np.arange(36,dtype=float).reshape(4,3,3);old[2,1,0]=np.nan
    np.savez_compressed(directory/'predictions.npz',model_selection_selected=old)
    return base,case,old

def test_timeout_attempt_keeps_exact_archive_and_explicit_state(tmp_path,monkeypatch):
    base,case,old=setup_paths(tmp_path,monkeypatch)
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('dummy',.01)
    monkeypatch.setattr(batch.subprocess,'run',timeout)
    result=batch.attempt(case,0,time.time()+2,{'compute_bound':{'per_case_timeout_seconds':1}})
    assert result['state']=='timeout'
    with np.load(base/'forecasts/development/toy/prediction.npz') as data:assert data['predicted'].tobytes()==old.tobytes()
    metadata=json.loads((base/'forecasts/development/toy/metadata.json').read_text());assert not metadata['switched'] and metadata['attempt_error']

def test_selected_full_failure_is_not_replaced_by_archive(tmp_path,monkeypatch):
    base,case,old=setup_paths(tmp_path,monkeypatch)
    folder=base/'forecasts/development/toy';folder.mkdir(parents=True)
    np.savez_compressed(folder/'prediction.npz',predicted=np.full_like(old,np.nan))
    (folder/'metadata.json').write_text(json.dumps({'case':case,'switched':True,'failed_full_frames':[0,1,2,3]}))
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    monkeypatch.setattr(batch,'fallback',lambda *a,**kw:pytest.fail('Selected future failure cannot trigger fallback'))
    result=batch.attempt(case,0,time.time()+2,{'compute_bound':{'per_case_timeout_seconds':1}})
    assert result['state']=='complete'
    with np.load(folder/'prediction.npz') as data:assert np.isnan(data['predicted']).all()

def test_terminal_gate_rejects_incomplete_before_input_or_evaluator_access(monkeypatch):
    monkeypatch.setattr(score,'guard',lambda:({'cases':[{}]*148},10))
    monkeypatch.setattr(score,'sha',lambda path:'digest')
    def read(path):
        if path.name=='method-freeze.json':return {}
        if path.name=='attempts.json':return {'protocol_sha256':'digest','method_freeze_sha256':'digest','cases':[]}
        pytest.fail('Gate must reject before any other reads')
    monkeypatch.setattr(score,'read',read)
    with pytest.raises(RuntimeError,match='All148'):score.terminal_gate()
