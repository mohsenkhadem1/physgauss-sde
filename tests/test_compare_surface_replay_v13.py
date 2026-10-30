import json,hashlib
import numpy as np
import pytest
from scripts.coupled_contact_v13 import compare_surface_replay as c

def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path)
 replay=tmp_path/'replay';selector='library/fem/0-0';old=tmp_path/'runs/coupled-surface-v13'/selector;new=replay/'outputs'/selector
 job={'path':selector};status=dict(job=job,simulation_settings={'dt':.01},protocol_sha256='protocol',success=True,diagnostics={'min_j':1.})
 arrays={k:np.zeros((2,3)) for k in ('frames','velocities','centers','times','reactions','rest','tets','tool_positions','active')}
 for path in (old,new):
  save(path/'complete.json',status);save(path/'diagnostics.json',status['diagnostics']);np.savez(path/'trajectory.npz',**arrays)
 request=dict(job=job,simulation_settings=status['simulation_settings'],protocol_sha256='protocol');save(replay/'replay-request.json',request)
 refresh(replay,new,status)
 return replay,old,new,status,arrays

def refresh(replay,new,status):
 result=dict(success=True,job=status['job'],simulation_settings=status['simulation_settings'],status=status,artifact_sha256={str((new/n).relative_to(replay)):sha(new/n) for n in ('complete.json','diagnostics.json','trajectory.npz')})
 save(replay/'replay-result.json',result);save(replay/'replay-execution.json',dict(returncode=0,timed_out=False,request_sha256=sha(replay/'replay-request.json'),replay_result_sha256=sha(replay/'replay-result.json')))

def test_exact_replay_passes(tmp_path,monkeypatch):
 replay,*_=fixture(tmp_path,monkeypatch);assert c.compare(replay)['passed']

@pytest.mark.parametrize('kind',['missing','nonfinite','changed_settings','failed_execution','wrong_request','tampered_result','both_missing'])
def test_bad_replay_rejected(tmp_path,monkeypatch,kind):
 replay,old,new,status,arrays=fixture(tmp_path,monkeypatch)
 if kind in ('missing','both_missing'):
  del arrays['frames'];np.savez(new/'trajectory.npz',**arrays)
  if kind=='both_missing':np.savez(old/'trajectory.npz',**arrays)
 elif kind=='nonfinite':
  arrays['frames'][0,0]=np.nan;np.savez(new/'trajectory.npz',**arrays)
 elif kind=='changed_settings':status['simulation_settings']['dt']=.02;save(new/'complete.json',status)
 refresh(replay,new,status)
 if kind=='failed_execution':
  execution=json.loads((replay/'replay-execution.json').read_text());execution['returncode']=1;save(replay/'replay-execution.json',execution)
 elif kind=='wrong_request':
  request=json.loads((replay/'replay-request.json').read_text());request['simulation_settings']['dt']=.7;save(replay/'replay-request.json',request)
 elif kind=='tampered_result':
  result=json.loads((replay/'replay-result.json').read_text());result['success']=False;save(replay/'replay-result.json',result)
 assert not c.compare(replay)['passed']
