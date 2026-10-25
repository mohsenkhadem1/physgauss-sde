import copy,json
import numpy as np
import pytest
from experiments import audit_fixture_balanced_nh_catalog_v01 as a
from tests.test_audit_fixture_balanced_held_nh_refinement_v01 import good_scalars


def surfaces():
    return {str(n):dict(passed=True,positive_quadrature_Gram_PSD=True,support=dict(normalization_error=1e-15,first_moment_error_m=1e-15),relative_support16_vs20=1e-15,maximum_load16_vs20_N=1e-15,refined_known_load_checks={l:dict(net_force_error_N=1e-15,moment_error_Nm=1e-15,normalization_error=1e-15,first_moment_error_m=1e-15)for l in a.stage.LOADS})for n in a.stage.LEVELS}


def rows():
    return [dict(n=n,label=l,completed=True,physical_passed=True,excitation_goal_met=False,legacy_load_eligible=False)for n in a.stage.LEVELS for l in a.stage.LOADS]


def test_all33_reference_validity_does_not_hide_weak_excitation():
    r=rows();spatial={l:dict(spatial_development_passed=True)for l in a.stage.LOADS};s=surfaces();out=a.group_classification(r,spatial,s)
    assert out==a.stage.group_classification(r,spatial,s)and out['all11_reference_valid']and not out['all_fields_excitation_goal_met']
    bad=copy.deepcopy(r);bad[0]['physical_passed']=False;assert not a.group_classification(bad,spatial,s)['all11_reference_valid']
    bad=copy.deepcopy(spatial);bad[a.stage.LOADS[0]]['spatial_development_passed']=False;assert not a.group_classification(r,bad,s)['all11_reference_valid']
    assert not a.group_classification(r[:-1],spatial,s)['all33_retained']


def test_legacy_prefix_exemption_preserved_without_reference_gate():
    row,d=good_scalars();d['nonrigid_RMS_m']=80e-6;row.update(excitation_goal_met=False,legacy_load_eligible=False,legacy_weak_prefix_control=False,nonrigid_RMS_goal_m=100e-6,nonrigid_fraction_goal=.3,label=a.stage.LOADS[1]);assert not a.field_status(row,d)['legacy_load_eligible']
    row.update(label='prefix-center-down',legacy_load_eligible=True,legacy_weak_prefix_control=True);assert a.field_status(row,d)['legacy_load_eligible']
    row['legacy_load_eligible']=False
    with pytest.raises(ValueError,match='legacy'):a.field_status(row,d)


def test_surface_false_pass_flags_and_missing_force_refused():
    s=surfaces();assert a.surfaces_pass(s)
    s['3']['maximum_load16_vs20_N']=1e-3
    with pytest.raises(ValueError,match='predicate'):a.surfaces_pass(s)
    s=surfaces();s['3']['refined_known_load_checks'].pop(a.stage.LOADS[0])
    with pytest.raises(ValueError,match='11'):a.surfaces_pass(s)


def test_independent_all11_spatial_formula_including_public_point_gate(monkeypatch):
    fields={(n,l):np.zeros((1,3))for n in a.stage.LEVELS for l in a.stage.LOADS}
    monkeypatch.setattr(a.stage.geometry,'common_rms',lambda fields,finest:np.array([[0,0,0],[100e-6,0,0],[0,5e-6,0]]));monkeypatch.setattr(a.stage.geometry,'field_queries',lambda n,u,p:np.full((len(p),3),n*10e-6))
    out=a.spatial_results(fields);assert out==a.stage.spatial_results(fields)and len(out)==11
    assert all(v['spatial_passed']and v['medium_adequate']and not v['spatial_development_passed']for v in out.values())


def fake_phase(monkeypatch,tmp_path):
    monkeypatch.setattr(a.stage,'LOCAL',tmp_path);monkeypatch.setattr(a.stage,'ROOT',tmp_path);monkeypatch.setattr(a.stage,'record',lambda p:dict(path=str(p),sha256=a.stage.sha(p)))
    (tmp_path/'release.json').write_text('{}');release=a.stage.record(tmp_path/'release.json');p=dict(source_SHA256={'test':'hash'},plan=dict(groups=a.stage.groups()));group=p['plan']['groups'][0];r=[]
    for n in a.stage.LEVELS:
        for l in a.stage.LOADS:
            path=tmp_path/'groups/group-00/raw-LOCAL'/f'n{n}-{l}.npz';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not-an-array');r.append(dict(n=n,label=l,completed=True,kind='NEW',group_index=0,layout=group['layout'],E_Pa=group['E_Pa'],K_N_m=group['K_N_m'],capture=a.stage.record(path)))
    gp=tmp_path/'groups/group-00/result-LOCAL.json';a.stage.write(gp,dict(completed=True,group=group,protocol_SHA256='proto',release=release,worker_pid=102,new_fields=33,reused_fields=0,wall_s=10.,peak_RSS_bytes=100,rows=r))
    cache=tmp_path/'runs/operator-dev/completed-metadata-cache-v01'/(a.stage.sha(gp)+'.json');cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(gp.read_bytes())
    wp=tmp_path/'worker-checkpoint-0.json';a.stage.write(wp,dict(completed=True,pid=102,slot=0,phase='checkpoint',protocol_SHA256='proto',release=release,groups=[dict(index=0,result=a.stage.record(gp),external_cache=a.stage.record(cache))],peak_RSS_bytes=100))
    e=dict(terminal=True,completed=True,phase='checkpoint',protocol_SHA256='proto',release=release,controller_pid=101,workers=[dict(pid=102,slot=0,groups=[0],exit_code=0,cleanup_exit_code=0,reaped=True,result=a.stage.record(wp))],peak_aggregate_RSS_bytes=200,wall_s=11.);ep=tmp_path/'execution-checkpoint.json';a.stage.write(ep,e)
    c=dict(root_exit_code=0,root_reaped=True,all_owned_PID_and_groups_absent=True,owned_pids=[101,102],protocol_SHA256='proto',source_SHA256=p['source_SHA256'],execution_SHA256=a.stage.sha(ep));a.stage.write(tmp_path/'checkpoint-root-closure.json',c)
    seen=[];monkeypatch.setattr(a.stage.controls.prior.shared.old,'absent',lambda pid:seen.append(pid));monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No NPZ before metadata barrier'))
    return p,c,r,seen


def test_phase_root_reap_context_and_every_capture_hash_before_arrays(monkeypatch,tmp_path):
    p,c,r,seen=fake_phase(monkeypatch,tmp_path);binding,g=a.phase_metadata('checkpoint',p,'proto');assert len(g)==1 and len(g[0]['value']['rows'])==33 and seen==[101,102]
    c['root_reaped']=False;a.stage.write(tmp_path/'checkpoint-root-closure.json',c)
    with pytest.raises(ValueError,match='closure'):a.phase_metadata('checkpoint',p,'proto')
    c['root_reaped']=True;a.stage.write(tmp_path/'checkpoint-root-closure.json',c)
    from pathlib import Path
    Path(r[-1]['capture']['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='capture'):a.phase_metadata('checkpoint',p,'proto')


def test_all8_and_summary_barrier_refuses_missing_group_before_npz(monkeypatch,tmp_path):
    monkeypatch.setattr(a.stage,'LOCAL',tmp_path);(tmp_path/'protocol.json').write_text('{}');p=dict(source_SHA256={a.SOURCE:'s',a.TEST:'t'})
    monkeypatch.setattr(a.stage,'validate',lambda d:p);monkeypatch.setattr(a.stage,'continuation_gate',lambda *a:None);monkeypatch.setattr(a,'phase_metadata',lambda phase,p,d:(dict(wall_s=1.),[dict(index=i)for i in([0]if phase=='checkpoint'else range(1,7))]));monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No arrays'))
    with pytest.raises(ValueError,match='All8'):a.metadata_barrier()


def test_simple_frozen_audit_resource_guards(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(a.os,'sched_getaffinity',lambda _: {16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setattr(a.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=100));monkeypatch.setattr(a.time,'monotonic',lambda:10.)
    a.guard(0.)
    monkeypatch.setattr(a.os,'sched_getaffinity',lambda _: {17})
    with pytest.raises(ValueError,match='CPU16'):a.guard(0.)
    monkeypatch.setattr(a.os,'sched_getaffinity',lambda _: {16});monkeypatch.setattr(a.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=4*1024**2))
    with pytest.raises(MemoryError):a.guard(0.)
    monkeypatch.setattr(a.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=100));monkeypatch.setattr(a.time,'monotonic',lambda:3600.)
    with pytest.raises(TimeoutError):a.guard(0.)
