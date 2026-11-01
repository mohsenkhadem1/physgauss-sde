import json
import numpy as np
from scripts.physical_refinement_v15 import world_render_bank as m
from scripts.physical_refinement_v15.observations import box_mesh

def test_world_resolution_paths_and_clean_geometry_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',tmp_path/'development');monkeypatch.setattr(m,'check_sources',lambda expected:None)
    folder=m.BASE/'worlds/development-2/n6/pose-000';folder.mkdir(parents=True);mesh=box_mesh([0,0,0],[.08,.06,.05],[.2,.5,.7]);path=folder/'sensor-geometry.npz';np.savez_compressed(path,**{'mesh0_'+k:v for k,v in mesh.items()});pose=np.zeros((2,6)).tolist()
    (folder/'complete.json').write_text(json.dumps(dict(pose_index=0,pose_m_rad=pose,artifact_sha256={'sensor-geometry.npz':m.sha(path)},hidden_E=999)))
    job=('development-2',6,'initial',0,pose);result=m.render_job(job,{},'2099-01-01T00:00:00Z');assert result['level']==6
    output=m.BASE/'world-render-bank/development-2/n6/initial';record=json.loads((output/'complete.json').read_text());assert record['level']==6 and 'hidden_E'not in record
    assert len(record['artifact_sha256'])==3
    with np.load(output/'left.npz')as a:assert set(a.files)=={'rgb','depth','features'}
    assert m.render_job(job,{},'2099-01-01T00:00:00Z')['cached']
