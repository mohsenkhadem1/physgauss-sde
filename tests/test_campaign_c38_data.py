import hashlib,json
import numpy as np
import pytest
from experiments.campaign_c38_data import load_public_geometry,load_public_rgbd_prefix,load_known_controls,load_qualified_targets
from experiments.campaign_static_p2_block import enrich


def record(tmp,name,**arrays):
    p=tmp/name;np.savez(p,**arrays);return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def geometry():
    X,e=enrich(np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),np.array([[0,1,2,3]]))
    return dict(rest_m=X,elements=e,tets=e[:,:4],faces=np.array([[0,1,2]]))


def test_public_members_and_prefix_only_time_barrier(tmp_path):
    g=geometry();r=record(tmp_path,'g.npz',**g);assert not load_public_geometry(tmp_path,r)['rest_m'].flags.writeable
    bad=record(tmp_path,'bad.npz',**g,material=np.ones(4))
    with pytest.raises(ValueError,match='allowlist'):load_public_geometry(tmp_path,bad)
    d=np.ones((2,1,2,2));rgb=np.zeros((*d.shape,3));v=np.ones_like(d,dtype=bool)
    r=record(tmp_path,'rgbd.npz',times_s=np.array([0.,.1]),rgb=rgb,depth_m=d,visible=v,camera_poses=np.tile(np.eye(4),(2,1,1,1)),intrinsics=np.tile(np.eye(3),(2,1,1,1)))
    assert load_public_rgbd_prefix(tmp_path,r,cutoff=.1)['information_role']=='public_acquired_RGBD_prefix_B'
    with pytest.raises(ValueError,match='future images'):load_public_rgbd_prefix(tmp_path,r,cutoff=.05)
    c=record(tmp_path,'controls.npz',force_times_s=np.array([0.,.1,.2]),force_coefficients=np.ones((2,1)),load_basis_N=np.zeros((1,10,3)))
    assert load_known_controls(tmp_path,c)['load_basis_N'].shape==(1,10,3)


def test_private_targets_require_admission_before_opening_missing_file(tmp_path):
    p=tmp_path/'gate.json';p.write_text(json.dumps(dict(qualified_for_prediction=False)))
    m=dict(domain_sha256='domain',forward_admission=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    # Missing raw record would raise FileNotFoundError if opened before admission.
    with pytest.raises(ValueError,match='unqualified'):load_qualified_targets(tmp_path,m,dict(path='DOES-NOT-EXIST.npz',sha256='0'*64))
    report=tmp_path/'evidence.json';report.write_text('{"manufactured_control":true}');source=tmp_path/'source.py';source.write_text('# synthetic')
    gate={k:True for k in ('qualified_for_prediction','spatial_qualified','temporal_qualified','nonlinear_solve_qualified','constitutive_matched')};gate.update(domain_sha256='domain',evidence=[dict(path=report.name,sha256=hashlib.sha256(report.read_bytes()).hexdigest())],source_sha256={source.name:hashlib.sha256(source.read_bytes()).hexdigest()})
    p.write_text(json.dumps(gate));m['forward_admission']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    g=geometry();r=record(tmp_path,'private.npz',**g,times_s=np.array([0.,.1]),positions_m=np.stack([g['rest_m']]*2),velocities_m_s=np.zeros((2,10,3)),constraint_mask=np.zeros((10,3),dtype=bool),private_extra_diagnostic=np.zeros(1))
    gate['admitted_trajectories']=[{**r,'case_id':'qualified-A','domain_sha256':'domain'}];r['case_id']='qualified-A';p.write_text(json.dumps(gate));m['forward_admission']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    q=load_qualified_targets(tmp_path,m,r);assert q.information_role.startswith('PRIVATE') and not q.positions_m.flags.writeable
    # Another readable, hash-valid trajectory cannot borrow A's admission.
    other=record(tmp_path,'unqualified-B.npz',**g,times_s=np.array([0.,.1]),positions_m=np.stack([g['rest_m']+1]*2),velocities_m_s=np.zeros((2,10,3)),constraint_mask=np.zeros((10,3),dtype=bool));other['case_id']='unqualified-B'
    with pytest.raises(ValueError,match='qualified domain'):load_qualified_targets(tmp_path,m,other)
    with pytest.raises(ValueError,match='identity'):load_qualified_targets(tmp_path,m,{**r,'case_id':'wrong-case'})
