import json,pickle
import numpy as np
import pytest
from scripts.compositional_dynamics_v17 import native_appearance_pilot as p

def test_shared_camera_mapping_and_known_geometry():
    assert [x['role']for x in p.camera_specs()]==['train']*4+['held']*2
    for spec in p.camera_specs():
        K,c2w=p.calibration(spec);eye,R=p.camera(spec).pose()
        for xyz in p.surface().reshape(-1,3)[::77]:
            q=R@(xyz-eye);q2=np.linalg.inv(c2w)@np.r_[xyz,1]
            assert np.allclose(q,q2[:3],atol=1e-14)
            uv=K@q2[:3];assert np.allclose(uv[:2]/uv[2],q[:2]/q[2]*spec['focal']+[424,240])
    assert np.allclose(np.ptp(p.surface().reshape(-1,3),axis=0),[.12,.06,.06])

def test_guard_rejects_held_images_in_train_catalog(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'BASE',tmp_path/'run');monkeypatch.setattr(p,'PLAN',tmp_path/'plan.json')
    folder=p.BASE/'train-input';folder.mkdir(parents=True)
    for i in range(4):
        (folder/f'{i}.png').write_bytes(b'fixture');(folder/f'mask_{i}.png').write_bytes(b'fixture')
    (folder/'camera_meta.pkl').write_bytes(pickle.dumps(dict(c2ws=[0]*4,intrinsics=[0]*4)))
    p.write(p.PLAN,dict(source_sha256={}))
    p.write(p.BASE/'input-manifest.json',dict(plan_sha256=p.sha(p.PLAN),files={str(f.relative_to(tmp_path)):p.sha(f)for f in folder.iterdir()}))
    p.guard();(folder/'held.png').write_bytes(b'forbidden')
    with pytest.raises(ValueError,match='Exact4'):p.guard()

def test_missing_depth_does_not_drop_failed_view():
    keys=['psnr_global_db','psnr_foreground_union_db','ssim_rgb_global','ssim_rgb_foreground_union','silhouette_iou','depth_joint_rmse_m']
    rows=[dict(role='held',metrics={k:1. for k in keys}),dict(role='held',metrics={k:None for k in keys})]
    result=p.summarize(rows)['held'];assert result['count']==2
    assert result['metrics']['depth_joint_rmse_m']==dict(mean=1.,available=1,expected=2)

@pytest.mark.parametrize('failure',['missing','tampered'])
def test_score_rejects_before_any_held_reference(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'BASE',tmp_path/'run');monkeypatch.setattr(p,'PLAN',tmp_path/'plan.json');p.BASE.mkdir()
    plan={'cameras':p.camera_specs()};p.write(p.PLAN,plan);monkeypatch.setattr(p,'guard',lambda:plan)
    p.write(p.BASE/'training-complete.json',{'complete':True});records=[]
    for c in plan['cameras']:
        f=tmp_path/f"{c['name']}.npz";f.write_bytes(b'fixture');records.append(dict(name=c['name'],role=c['role'],path=f.name,sha256=p.sha(f)))
    if failure=='missing':records=records[:-1]
    else:(tmp_path/records[-1]['path']).write_bytes(b'mutated')
    p.write(p.BASE/'prediction-commit.json',dict(plan_sha256=p.sha(p.PLAN),training_complete_sha256=p.sha(p.BASE/'training-complete.json'),records=records))
    def forbidden(*args):raise AssertionError('Future reference accessed before guard')
    monkeypatch.setattr(p,'reference',forbidden)
    with pytest.raises(ValueError,match='Exact6|Prediction changed'):p.score()
