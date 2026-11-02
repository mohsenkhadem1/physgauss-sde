import sys,time,json
from pathlib import Path
from scripts.compositional_dynamics_v17 import native_cohort_recovery as r

def test_stage_passes_explicit_environment_and_private_output(tmp_path):
    result=r.stage([sys.executable,'-c',"import os,json;print(json.dumps({k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','PYNPUT_BACKEND']}))"],tmp_path/'stage',10,time.time()+20)
    assert result['completed']
    assert json.loads((tmp_path/'stage/output.log').read_text())=={'CUDA_VISIBLE_DEVICES':'0','PYNPUT_BACKEND':'dummy'}

def test_expired_deadline_does_not_spawn(tmp_path):
    result=r.stage([sys.executable,'-c',"raise RuntimeError('should never run')"],tmp_path/'stage',10,time.time()-1)
    assert not result['completed']and 'pid'not in result
    assert 'deadline' in result['error']

def test_gpu_headroom_matches_original_threshold(monkeypatch):
    responses=iter(['6143, 0','6144, 91','6144, 90'])
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:next(responses))
    monkeypatch.setattr(r.time,'sleep',lambda value:None)
    result=r.gpu_ready(time.time()+10)
    assert result['free_MiB']==6144 and result['utilization_percent']==90
