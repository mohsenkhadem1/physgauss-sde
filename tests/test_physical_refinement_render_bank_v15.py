import json
import numpy as np
import pytest
from scripts.physical_refinement_v15 import render_bank as m
from scripts.physical_refinement_v15.observations import box_mesh

def geometry(path):
    mesh=box_mesh([0,0,0],[.08,.06,.05],[.2,.5,.7]);np.savez_compressed(path,**{'mesh0_'+k:v for k,v in mesh.items()})

def test_geometry_only_contract_and_feature_shape(tmp_path):
    p=tmp_path/'geometry.npz';geometry(p);image,mean=m.render_geometry(p,m.Camera((0,-.3,0),(0,0,0),256,192,250))
    assert mean.shape==(64,) and np.isfinite(mean).all()
    with np.load(p)as a:arrays={k:a[k]for k in a.files}
    arrays['hidden_stiffness']=12000;np.savez_compressed(p,**arrays)
    with pytest.raises(ValueError,match='Only complete geometry'):m.render_geometry(p,m.CAMERAS['left'])

def test_cache_rejects_changed_geometry_and_preserves_clean_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',tmp_path/'development');monkeypatch.setattr(m,'check_sources',lambda expected:None)
    folder=m.BASE/'hypotheses/case/n4/pose-000';folder.mkdir(parents=True);geometry(folder/'sensor-geometry.npz');pose=np.zeros((2,6)).tolist();record=dict(pose_index=0,pose_m_rad=pose,artifact_sha256={'sensor-geometry.npz':m.sha(folder/'sensor-geometry.npz')},private_material_label=12000)
    (folder/'complete.json').write_text(json.dumps(record));job=('case','initial',0,pose)
    m.render_job(job,{},'2099-01-01T00:00:00Z');output=m.BASE/'render-bank/case/initial'
    with np.load(output/'left.npz')as a:assert set(a.files)=={'rgb','depth','features'}
    completion=json.loads((output/'complete.json').read_text());assert 'private_material_label'not in completion
    assert m.render_job(job,{},'2099-01-01T00:00:00Z')['cached']
    with open(folder/'sensor-geometry.npz','ab')as f:f.write(b'changed')
    with pytest.raises(ValueError):m.render_job(job,{},'2099-01-01T00:00:00Z')
