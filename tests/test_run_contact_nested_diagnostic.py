import json
import numpy as np
import pytest
from experiments import run_contact_nested_diagnostic as m


def test_prepare_freezes_all_existing_clocks_without_saved_array_calls(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE','base.json');monkeypatch.setattr(m,'SUMMARY','summary.json');monkeypatch.setattr(m,'sources',lambda:['source.py']);(tmp_path/'source.py').write_text('source snapshot');(tmp_path/'capture.npz').write_bytes(b'only-hashed-not-parsed')
 (tmp_path/'base.json').write_text(json.dumps(dict(outputs_s=[i/120 for i in range(10,19)])))
 (tmp_path/'summary.json').write_text(json.dumps(dict(completed=True,passed=True,capture=dict(path='capture.npz',sha256=m.sha(tmp_path/'capture.npz')))))
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Saved array during prepare')))
 digest=m.prepare('protocol.json','output');p=json.loads((tmp_path/'protocol.json').read_text())
 assert p['clocks_s']==[i/120 for i in range(10,19)]and p['physics']['kappa_N_m3']==1e9 and p['resources']==m.RESOURCES
 assert digest==m.sha(tmp_path/'protocol.json')and not(tmp_path/'output').exists()
 with pytest.raises(FileExistsError):m.prepare('protocol.json','output')


def test_failed_partition_proof_blocks_saved_deformation_access(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'validate',lambda*a:dict(output='out',coarse_cells=[1,1,1],fine_cells=[2,2,2]));monkeypatch.setattr(m,'guard',lambda:None)
 monkeypatch.setattr(m.control,'nested_transfer',lambda*a:(_ for _ in()).throw(ValueError('Crossing coarse partition')))
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Saved field before nesting proof')))
 result=m.run('protocol','hash');assert not result['completed']and 'Crossing coarse partition'in result['error']and result['new_trajectory_calls']==0
