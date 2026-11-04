import json
from pathlib import Path
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.isaac_prefix_inputs import export_prefix


def fixture(root):
    root.mkdir();nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    ref=dict(rest_points_m=nodes,tetrahedra=np.array([[0,1,2,3]]),surface_faces=np.array([[0,1,2]]),vertex_colors=np.ones((4,3)),means_m=np.array([[1/3,1/3,0]]),colors=np.ones((1,3)),scales_m=np.ones((1,3))*.01,opacities=np.ones(1),face_vertex_ids=np.array([[0,1,2]]),barycentric_weights=np.ones((1,3))/3)
    np.savez(root/'reference.npz',**ref)
    obs=root/'observations';obs.mkdir();cam=dict(K=np.eye(3),frame_ids=np.array([0,10,20,40]),times_s=np.array([0.,.5,1.,2.]),camera_to_world_usd_row=np.tile(np.eye(4),(4,1,1)))
    np.savez(obs/'cameras.npz',**cam)
    for sensor,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
        (obs/sensor).mkdir()
        for frame in [0,10,20]: (obs/sensor/f'{frame:05d}{suffix}').write_bytes(b'fixture bytes, not decoded')
    return ref,cam


def test_prefix_private_and_future_changes_do_not_affect_package(tmp_path):
    source=tmp_path/'source';ref,cam=fixture(source)
    first=export_prefix(source,tmp_path/'first')
    assert first['observed_frames']==[0,10,20]
    (source/'truth').mkdir();(source/'truth/trajectory.npz').write_bytes(b'forbidden')
    (source/'summary.json').write_text('forbidden');(source/'recipe.json').write_text('forbidden')
    cam['camera_to_world_usd_row'][-1]+=100
    np.savez(source/'observations/cameras.npz',**cam)
    for sensor,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
        (source/'observations'/sensor/f'00040{suffix}').write_bytes(b'future secret')
    second=export_prefix(source,tmp_path/'second')
    assert first==second
    assert not first['known_position_commands'] and not first['force_labels_exported']
    with pytest.raises(FileExistsError):export_prefix(source,tmp_path/'first')


def test_forbidden_array_field_rejected(tmp_path):
    source=tmp_path/'source';ref,_=fixture(source)
    np.savez(source/'reference.npz',**ref,forces=np.zeros((3,)))
    with pytest.raises(ValueError,match='Unexpected'):export_prefix(source,tmp_path/'out')
    assert not(tmp_path/'out').exists()


def test_missing_future_files_allowed_but_prefix_files_required(tmp_path):
    source=tmp_path/'source';fixture(source)
    export_prefix(source,tmp_path/'valid')
    (source/'observations/rgb/00010.png').unlink()
    with pytest.raises(ValueError,match='Missing'):export_prefix(source,tmp_path/'invalid')
