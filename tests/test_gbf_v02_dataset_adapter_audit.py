"""Independent contract tests: suffix perturbation, corrected masks, fresh guard."""
import json,pickle
from pathlib import Path
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import benchmark_inputs as api


def phys_fixture(root):
    folder=root/'single_lift_test';folder.mkdir()
    rest=np.array([[x,y,z]for x in (0.,.1)for y in (0.,.1)for z in (0.,.1)])
    x=np.stack([rest+np.array([.001*t,0,0])for t in range(10)])
    ctrl=np.stack([rest[:3]+np.array([.002*t,0,0])for t in range(10)])
    raw=dict(object_points=x,object_visibilities=np.ones((10,8),bool),
             object_motions_valid=np.ones((10,8),bool),controller_points=ctrl,
             surface_points=rest.copy(),interior_points=rest.mean(0)[None],
             forces=np.ones((10,3))*123)
    (folder/'split.json').write_text(json.dumps({'train':[0,6],'test':[6,10],'frame_len':10}))
    write_raw(folder,raw)
    with (folder/'gt_track_3d.pkl').open('wb')as f:pickle.dump(x,f)
    return folder,raw


def write_raw(folder,raw):
    with (folder/'final_data.pkl').open('wb')as f:pickle.dump(raw,f)


def test_phystwin_suffix_isolated_and_correct_incoming_mask(tmp_path):
    folder,raw=phys_fixture(tmp_path)
    original=api.load_phystwin_predictor(folder.name,max_nodes=8,root=tmp_path)
    digest=api.predictor_digest(original)
    raw['object_points'][6:]+=1000
    raw['object_visibilities'][6:]=False
    raw['object_motions_valid'][5:]=False # t5 outgoing must not affect observed t5.
    write_raw(folder,raw)
    with(folder/'gt_track_3d.pkl').open('wb')as f:pickle.dump(np.zeros((10,8,3)),f)
    changed=api.load_phystwin_predictor(folder.name,max_nodes=8,root=tmp_path)
    assert api.predictor_digest(changed)==digest
    assert changed['observed'].shape==(6,8,3)
    assert changed['controller_targets'].shape[0]==10
    assert changed['mask_version']=='incoming-v22'
    assert 'forces' not in changed and 'evaluation'not in changed
    raw['object_motions_valid'][1,0]=False;write_raw(folder,raw)
    changed=api.load_phystwin_predictor(folder.name,max_nodes=8,root=tmp_path)
    assert changed['observed_mask'][1,0] and not changed['observed_mask'][2,0]


def test_phystwin_rgb_paths_and_cutoff_are_prefix_only(tmp_path):
    folder,_=phys_fixture(tmp_path)
    data=api.load_phystwin_predictor(folder.name,max_nodes=8,prefix_end=3,include_rgb_paths=True,root=tmp_path)
    assert len(data['prefix_rgb_paths'])==3
    assert all(Path(p).stem in ('0','1','2')for row in data['prefix_rgb_paths']for p in row)
    assert len(data['times'])==10
    with pytest.raises(ValueError):api.load_phystwin_predictor(folder.name,prefix_end=7,root=tmp_path)
    with pytest.raises(ValueError):api.load_phystwin_predictor('../bad',root=tmp_path)


def soft_fixture(root):
    h5py=pytest.importorskip('h5py')
    p=root/'object-soft/libero_object/libero_object_task1/replayed_demos/test.hdf5';p.parent.mkdir(parents=True)
    rest=np.array([[x,y,z]for x in (0.,.04)for y in (0.,.04)for z in (0.,.04)])
    x=np.stack([rest+np.array([.001*t,0.,0.])for t in range(10)])
    pose=np.tile(np.r_[rest.mean(0),1.,0.,0.,0.],(10,1))
    with h5py.File(p,'w')as f:
        g=f.create_group('data/demo_0');g.attrs['asset_name']='toy'
        g.create_dataset('states/deformable_object/toy/nodal_pos_w',data=x)
        g.create_dataset('obs/eef_pose',data=pose)
        g.create_dataset('obs/gripper_width',data=np.ones(10)*.02)
        g.create_dataset('obs/gripper_binary',data=-np.ones(10))
        g.create_dataset('obs/gripper_net_force',data=np.ones((10,3))*42)
        g.create_dataset('obs/fem_deformation_rms',data=np.ones(10)*999)
        g.create_dataset('initial_state/articulation/robot/root_pose',data=np.array([[0.,0.,0.,1.,0.,0.,0.]]))
    return p


def test_softvt_only_reads_prefix_and_never_force_channels(tmp_path):
    h5py=pytest.importorskip('h5py');p=soft_fixture(tmp_path)
    a=api.load_softvt_predictor('object-soft',1,0,root=tmp_path)
    with h5py.File(p,'a')as f:
        f['data/demo_0/states/deformable_object/toy/nodal_pos_w'][6:]=1000
        f['data/demo_0/obs/gripper_net_force'][:]=-999
        f['data/demo_0/obs/fem_deformation_rms'][:]=-999
    b=api.load_softvt_predictor('object-soft',1,0,root=tmp_path)
    assert api.predictor_digest(a)==api.predictor_digest(b)
    assert len(b['observed'])==6 and len(b['times'])==10
    assert 'gripper_net_force'not in b and 'fem_deformation_rms'not in b
    evaluation=api.load_softvt_evaluation('object-soft',1,0,root=tmp_path)
    assert np.all(evaluation['points'][6:]==1000)


def test_fresh_softvt_is_rejected_before_any_file_access(tmp_path):
    for task,episode in [(0,10),(1,50),(2,99)]:
        with pytest.raises(PermissionError):api.load_softvt_predictor('object-soft',task,episode,root=tmp_path)
        with pytest.raises(PermissionError):api.load_softvt_evaluation('spatial-soft',task,episode,root=tmp_path)


def test_predictor_validation_rejects_evaluator_fields(tmp_path):
    folder,_=phys_fixture(tmp_path)
    data=api.load_phystwin_predictor(folder.name,max_nodes=8,root=tmp_path)
    with pytest.raises(ValueError):api.validate_predictor(dict(data,evaluation={}))
    with pytest.raises(ValueError):api.validate_predictor(dict(data,observed=data['observed'][:-1]))
