import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments import video_newton_reference as video

def fixture(root):
    def write(name,data):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));return p
    def rec(name):return dict(path=name,sha256=video.sha(root/name))
    for name in ('gpu-source.py','cpu-source.py'):(root/name).write_text('# synthetic only\n')
    X=np.array([[0.,0,0],[.2,0,0],[0,.08,0],[0,0,.08]]);T=np.array([[0,1,2,3]]);F=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    q=np.repeat(X[None],25,axis=0);gq=q.copy();gq[:,1,0]+=np.linspace(0,1e-6,25)
    mask=np.zeros_like(X,dtype=bool);mask[0]=True
    shared=dict(times_s=np.arange(25)/240,query_times_s=np.arange(25)/240,material_probes_m=X[:2],rest_m=X,tets=T,faces=F,physical_masses_kg=np.ones(4),applied_forces_N=np.zeros_like(X))
    (root/'cpu').mkdir();(root/'gpu').mkdir()
    np.savez(root/'cpu/record.npz',positions_m=q,query_positions_m=q[:,:2],constraint_mask=mask,**shared)
    np.savez(root/'gpu/record.npz',positions_m=gq,query_positions_m=gq[:,:2],fixed_node_ids=np.array([0]),**shared)
    cpu=dict(method='BDF2',hz=240,completed=True,capture=rec('cpu/record.npz'),maximum_physical_original_free_R_N=1e-12,minimum_J=1.,maximum_constraint_defect_m=0.)
    gpu=dict(method='BDF2',hz=240,iterations=128,completed=True,passed=True,capture=rec('gpu/record.npz'),maximum_CPU_allnode_position_error_m=float(np.linalg.norm(gq-q,axis=-1).max()),maximum_CPU_fixedquery_position_error_m=float(np.linalg.norm(gq[:,:2]-q[:,:2],axis=-1).max()))
    write('cpu/case.json',cpu);write('gpu/case.json',gpu)
    write('cpu/protocol.json',dict(source_sha256={'cpu-source.py':video.sha(root/'cpu-source.py')}))
    write('gpu/protocol.json',dict(source_sha256={'gpu-source.py':video.sha(root/'gpu-source.py')},external_source_sha256={},CPU_protocol_sha256={'cpu/protocol.json':video.sha(root/'cpu/protocol.json')},CPU_reference_sha256={p:video.sha(root/p) for p in ('cpu/record.npz','cpu/case.json')},jobs=[dict(method='BDF2',hz=240,iterations=128)],nonclaims=['Synthetic renderer test only']))
    write('cpu/summary.json',dict(completed=True,protocol=rec('cpu/protocol.json'),cases=[cpu]));write('gpu/summary.json',dict(completed=True,protocol=rec('gpu/protocol.json'),cases=[gpu]))
    return dict(gpu_case='gpu/case.json',gpu_capture='gpu/record.npz',gpu_summary='gpu/summary.json',gpu_protocol='gpu/protocol.json',gpu_protocol_sha256=video.sha(root/'gpu/protocol.json'),cpu_case='cpu/case.json',cpu_capture='cpu/record.npz',cpu_summary='cpu/summary.json',cpu_protocol='cpu/protocol.json',cpu_protocol_sha256=video.sha(root/'cpu/protocol.json'),root=root)

def test_explicit_case_pair_and_recorded_hold_schedule(tmp_path):
    args=fixture(tmp_path);pair=video.load_pair(**args)
    assert pair['checked'];np.testing.assert_array_equal(pair['sample_ids'],np.arange(25))
    ids,slow=video.schedule();assert len(ids)==115 and slow==32 and np.all(ids[:4]==0) and np.all(ids[-19:]==24)
    assert set(ids)==set(range(25))
    with pytest.raises(ValueError):video.schedule(repeats=8)

def test_hash_and_completion_guards_before_state_reads(tmp_path,monkeypatch):
    args=fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('State arrays opened before completion/source gates')
    monkeypatch.setattr(video.np,'load',forbidden)
    p=tmp_path/'gpu/summary.json';data=json.loads(p.read_text());data['completed']=False;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='completed'):video.load_pair(**args)
    data['completed']=True;p.write_text(json.dumps(data));(tmp_path/'gpu-source.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed hash'):video.load_pair(**args)

def test_complete_failed_case_requires_explicit_failure_label(tmp_path):
    args=fixture(tmp_path);p=tmp_path/'gpu/case.json';g=json.loads(p.read_text());g['passed']=False;p.write_text(json.dumps(g))
    p=tmp_path/'gpu/summary.json';s=json.loads(p.read_text());s['cases']=[g];p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='allow_failed'):video.load_pair(**args)
    assert not video.load_pair(**args,allow_failed=True)['checked']

def test_time_roundoff_allowed_but_changed_clock_rejected(tmp_path):
    args=fixture(tmp_path)
    def replace_times(offset):
        path=tmp_path/'gpu/record.npz'
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        arrays['times_s']=np.arange(25)/240+offset;arrays['query_times_s']=np.arange(25)/240+offset
        np.savez(path,**arrays)
        case=tmp_path/'gpu/case.json';g=json.loads(case.read_text());g['capture']['sha256']=video.sha(path);case.write_text(json.dumps(g))
        summary=tmp_path/'gpu/summary.json';s=json.loads(summary.read_text());s['cases']=[g];summary.write_text(json.dumps(s))
    replace_times(np.spacing(.1))
    assert video.load_pair(**args)['checked']
    replace_times(1e-6)
    with pytest.raises(AssertionError):video.load_pair(**args)
