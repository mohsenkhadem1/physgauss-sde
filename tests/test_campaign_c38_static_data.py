import hashlib
import numpy as np
import pytest
from experiments.campaign_c38_static_data import load_public_static_controls,load_public_static_rgbd_prefix


def save(tmp_path,**members):
    p=tmp_path/'data.npz';np.savez(p,**members)
    return dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def frames():
    return dict(acquisition_indices=np.array([0,2]),load_state_indices=np.array([10,10]),rgb=np.zeros((2,1,2,3,3)),depth_m=np.ones((2,1,2,3)),visible=np.ones((2,1,2,3),bool),camera_poses=np.broadcast_to(np.eye(4),(2,1,4,4)),intrinsics=np.broadcast_to(np.eye(3),(2,1,3,3)))


def test_static_controls_are_not_timed_or_material_labeled(tmp_path):
    r=save(tmp_path,load_state_indices=np.array([10,11]),load_vectors_N=np.zeros((2,4,3)))
    result=load_public_static_controls(tmp_path,r,number_of_nodes=4)
    assert 'times_s' not in result and result['load_vectors_N'].shape==(2,4,3)
    r=save(tmp_path,load_state_indices=np.array([10,11]),load_vectors_N=np.zeros((2,4,3)),young_Pa=np.array(100000.))
    with pytest.raises(ValueError,match='allowlist'):load_public_static_controls(tmp_path,r,number_of_nodes=4)


def test_equilibrium_prefix_allows_repeated_state_but_no_future_or_private_members(tmp_path):
    data=frames();r=save(tmp_path,**data)
    result=load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[10,11])
    assert result['load_state_indices'].tolist()==[10,10] and 'velocities_m_s' not in result
    with pytest.raises(ValueError,match='Only acquired'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=1,known_load_state_indices=[10,11])
    with pytest.raises(ValueError,match='declared public'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[11])
    for key in ('times_s','constraint_mask','positions_m','young_Pa'):
        r=save(tmp_path,**data,**{key:np.array(1.)})
        with pytest.raises(ValueError,match='allowlist'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[10])


def test_unacquired_images_are_not_accessed(tmp_path,monkeypatch):
    data=frames();r=save(tmp_path,**data);calls=[];original=np.lib.npyio.NpzFile.__getitem__
    def track(self,key):calls.append(key);return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',track)
    with pytest.raises(ValueError,match='Only acquired'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=1,known_load_state_indices=[10])
    assert calls==['acquisition_indices']


def test_hash_and_missing_depth_convention(tmp_path):
    data=frames();data['visible'][0,0,0,0]=False;r=save(tmp_path,**data)
    with pytest.raises(ValueError,match='zero missing'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[10])
    data['depth_m'][0,0,0,0]=0;r=save(tmp_path,**data)
    load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[10])
    r['sha256']='0'*64
    with pytest.raises(ValueError,match='Changed/outside'):load_public_static_rgbd_prefix(tmp_path,r,last_acquisition=2,known_load_state_indices=[10])
