import json
from scripts.gaussian_boundary_flow_v02 import expansion_downstream as module


def test_worker_failure_and_orphan_are_terminal_distinctions(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'BASE',tmp_path)
    assert module.source_state('case',[{'pid':1,'cases':['case']}])[0]=='waiting'
    assert module.source_state('case',[])[0]=='orphaned'
    worker=tmp_path/'worker-cuda-0';worker.mkdir()
    (worker/'status.json').write_text(json.dumps({'completed_jobs':[{'case':'case','state':'failed','returncode':1}]}))
    assert module.source_state('case',[])[0]=='failed'


def test_completed_source_requires_valid_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'BASE',tmp_path);source=tmp_path/'case';source.mkdir();(source/'complete.json').write_text('{}')
    def reject(*args):raise ValueError('tampered')
    monkeypatch.setattr(module,'load_sealed',reject)
    state,error=module.source_state('case',[])
    assert state=='failed' and error['reason']=='source_seal_invalid'
    monkeypatch.setattr(module,'load_sealed',lambda *args:None)
    assert module.source_state('case',[])[0]=='ready'
