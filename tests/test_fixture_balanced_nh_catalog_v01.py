import copy,inspect,json,os,signal,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import run_fixture_balanced_nh_catalog_v01 as m


def test_all264_interior_memberships_no_reuse_no_hidden_queue():
    groups=m.groups();rows=m.worklist();assert len(groups)==8 and len(rows)==264 and len({v['id']for v in rows})==264
    assert [g['index']for g in groups]==list(range(8))and{g['E_Pa']for g in groups}=={2500.,6500.}
    assert all(g['K_N_m']==float(np.sqrt(150.))for g in groups)and len({g['layout']for g in groups})==4
    assert all(sum(v['group_index']==g['index']for v in rows)==33 for g in groups)
    assert all(v['kind']=='NEW'and v['status']=='queued'for v in rows)
    assert m.assignments('checkpoint')==[[0]]and sorted(sum(m.assignments('remaining'),[]))==list(range(1,8))
    with pytest.raises(ValueError):m.assignments('all')
    p=m.plan();assert p['gates']==m.controls.metrics.GATES and p['fields']==264 and p['reused_fields']==0 and p['group_load_references']==88
    assert m.cost_projection()['new_rest_factors']==24 and m.cost_projection()['estimated_group_compute_s']<5400


def test_reference_valid_excludes_excitation_but_legacy_flags_preserved():
    d=dict(minimum_J=.98,maximum_absolute_symmetric_strain_eigenvalue=.02,nonrigid_RMS_m=80e-6,nonrigid_fraction=.6)
    a=m.field_status(d,1e-12,True,dict(passed=True),dict(passed=True),'prefix-center-down')
    b=m.field_status(d,1e-12,True,dict(passed=True),dict(passed=True),m.LOADS[1])
    assert a['physical_passed']and not a['excitation_goal_met']and a['legacy_load_eligible']and a['legacy_weak_prefix_control']
    assert b['physical_passed']and not b['legacy_load_eligible']and not b['legacy_weak_prefix_control']
    rows=[dict(n=n,label=l,completed=True,physical_passed=True,excitation_goal_met=False)for n in m.LEVELS for l in m.LOADS]
    spatial={l:dict(spatial_development_passed=True)for l in m.LOADS};surface={str(n):dict(passed=True)for n in m.LEVELS}
    c=m.group_classification(rows,spatial,surface);assert c['all11_reference_valid']and not c['all_fields_excitation_goal_met']and not c['bank_admitted']
    bad=copy.deepcopy(rows);bad[0]['physical_passed']=False;assert not m.group_classification(bad,spatial,surface)['all11_reference_valid']
    spatial[m.LOADS[-1]]['spatial_development_passed']=False;assert not m.group_classification(rows,spatial,surface)['all11_reference_valid']
    assert not m.group_classification(rows[:-1],spatial,surface)['all33_retained']


def test_each_layout_same_knownforces_and_actual_PSDsurface_guards():
    first=None
    for layout in m.bank.LAYOUTS:
        # Manufactured coarse geometry/quadrature only; no equilibrium.
        f=m.geometry.make_forward(1,layout['id']);loads,info=m.bank.load_catalog(1)
        a=m.surface_checks(f,loads,layout['id']);assert a['passed']and a['positive_quadrature_Gram_PSD']
        if first is not None:assert all(np.array_equal(loads[k],first[k])for k in m.LOADS)
        first=loads
    assert len(loads)==11 and m.bank.measurement_points().shape==(16,3)


def test_original_query_spatial_threshold_all11_no_excitation_filter(monkeypatch):
    fields={(n,l):np.zeros((1,3))for n in m.LEVELS for l in m.LOADS}
    monkeypatch.setattr(m.geometry,'common_rms',lambda fields,finest:np.array([[0,0,0],[100e-6,0,0],[0,5e-6,0]]))
    monkeypatch.setattr(m.geometry,'field_queries',lambda n,u,p:np.full((len(p),3),n*10e-6))
    values=m.spatial_results(fields);assert len(values)==11
    assert all(v['spatial_passed']and v['medium_adequate']and not v['public_point20um_passed']and not v['spatial_development_passed']for v in values.values())


def test_exact_external_group_cache_never_overwrites_changed_bytes(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)))
    p=tmp_path/'group.json';p.write_text('{"complete":true}');r=m.record(p);cache=m.cache_metadata(r)
    assert Path(cache['path']).read_bytes()==p.read_bytes()and m.cache_metadata(r)==cache
    Path(cache['path']).write_text('changed')
    with pytest.raises(ValueError,match='cache'):m.cache_metadata(r)


def fake_terminal(monkeypatch,tmp_path,first_reference_valid=True):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)))
    for name in('protocol.json','release.json'):(tmp_path/name).write_text('{}')
    audit_pins={}
    for name in(m.AUDIT_SOURCE,m.AUDIT_TEST):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic independent procedure');audit_pins[name]=m.sha(path)
    digest=m.sha(tmp_path/'protocol.json');p=dict(plan=m.plan(),source_SHA256=dict({'synthetic':'source'},**audit_pins));monkeypatch.setattr(m,'validate',lambda d:p);monkeypatch.setattr(m.controls.prior.shared.old,'absent',lambda pid:None)
    def verify(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=set(value)and m.record(m.ROOT/value['path'])!=value:raise ValueError('Changed synthetic commitment')
            for v in value.values():verify(v)
        elif isinstance(value,list):
            for v in value:verify(v)
    monkeypatch.setattr(m.controls.screen,'verify_records',verify)
    release=m.record(tmp_path/'release.json');grouprecords={}
    for group in m.groups():
        directory=tmp_path/'groups'/f'group-{group["index"]:02d}';directory.mkdir(parents=True);rows=[]
        for n in m.LEVELS:
            for label in m.LOADS:
                capture=directory/'raw-LOCAL'/f'n{n}-{label}.npz';capture.parent.mkdir(exist_ok=True);capture.write_bytes(b'NO ARRAY')
                rows.append(dict(n=n,label=label,group_index=group['index'],layout=group['layout'],E_Pa=group['E_Pa'],K_N_m=group['K_N_m'],completed=True,kind='NEW',physical_passed=True,excitation_goal_met=False,capture=m.record(capture)))
        surface={str(n):dict(surface=dict(passed=True))for n in m.LEVELS};spatial={l:dict(spatial_development_passed=True)for l in m.LOADS}
        if group['index']==0 and not first_reference_valid:spatial[m.LOADS[0]]['spatial_development_passed']=False
        flags=m.group_classification(rows,spatial,{n:v['surface']for n,v in surface.items()})
        value=dict(completed=True,group=group,protocol_SHA256=digest,release=release,rows=rows,meshes=surface,spatial=spatial,new_fields=33,reused_fields=0,wall_s=10.,peak_RSS_bytes=100,**flags)
        grouprecords[group['index']]=(directory/'result-LOCAL.json',value)
    for phase in('checkpoint','remaining'):
        e=dict(completed=True,terminal=True,phase=phase,controller_pid=100 if phase=='checkpoint'else 200,protocol_SHA256=digest,release=release,workers=[],wall_s=20.,peak_aggregate_RSS_bytes=100)
        for slot,ids in enumerate(m.assignments(phase)):
            pid=e['controller_pid']+slot+1;entries=[]
            for i in ids:
                path,g=grouprecords[i];g['worker_pid']=pid;m.write(path,g);rec=m.record(path);entries.append(dict(index=i,result=rec,external_cache=m.cache_metadata(rec)))
            target=tmp_path/f'worker-{phase}-{slot}.json';w=dict(completed=True,pid=pid,slot=slot,phase=phase,protocol_SHA256=digest,release=release,groups=entries,queue=[dict(index=i,status='completed')for i in ids],peak_RSS_bytes=100);m.write(target,w)
            e['workers'].append(dict(pid=pid,slot=slot,groups=ids,reaped=True,exit_code=0,cleanup_exit_code=0,result=m.record(target)))
        target=tmp_path/f'execution-{phase}.json';m.write(target,e)
        closure=dict(root_reaped=True,root_exit_code=0,all_owned_PID_and_groups_absent=True,protocol_SHA256=digest,execution_SHA256=m.sha(target),source_SHA256=p['source_SHA256'],owned_pids=[e['controller_pid'],*[w['pid']for w in e['workers']]])
        m.write(tmp_path/f'{phase}-root-closure.json',closure)
    checkpoint=m.checkpoint_gate(p,digest);m.write(tmp_path/'continuation-release.json',dict(approved=True,protocol_SHA256=digest,source_SHA256=p['source_SHA256'],checkpoint=checkpoint,resources=m.LIMITS))
    return p,digest,grouprecords


def test_all264_hashes_and_8groups_before_any_array_or_downstream_gate(monkeypatch,tmp_path):
    p,digest,records=fake_terminal(monkeypatch,tmp_path);monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No actual arrays in terminalmetadata'))
    _,phases,groups=m.terminal_records(digest);assert len(groups)==8 and sum(len(v['value']['rows'])for v in groups)==264
    rec=m.finalize(digest);s=json.loads(Path(rec['path']).read_text());assert s['all88_reference_valid']and not s['all_fields_excitation_goal_met']
    with pytest.raises(FileNotFoundError):m.completed_catalog_gate()
    last=records[7][1]['rows'][-1]['capture'];Path(last['path']).write_bytes(b'TAMPER')
    with pytest.raises(ValueError,match='capture'):m.terminal_records(digest)


def test_no_continuation_without_exact_costcheckpoint_rootclosure(monkeypatch,tmp_path):
    p,digest,_=fake_terminal(monkeypatch,tmp_path);m.continuation_gate(p,digest)
    path=tmp_path/'continuation-release.json';v=json.loads(path.read_text());v['approved']=False;m.write(path,v)
    with pytest.raises(ValueError,match='continuationrelease'):m.continuation_gate(p,digest)
    c=tmp_path/'checkpoint-root-closure.json';v=json.loads(c.read_text());v['root_reaped']=False;m.write(c,v)
    with pytest.raises(ValueError,match='closure'):m.checkpoint_gate(p,digest)


def test_owned_stop_cleanup_signal_restoration_without_group_calls(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda *a:None);monkeypatch.setattr(m,'validate',lambda d:{});monkeypatch.setattr(m,'available_memory',lambda:200*1024**3);monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256='fake'));(tmp_path/'release.json').write_text('{}')
    before={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT)};cleaned=[]
    class Child:
        pid=999999;returncode=-15
        def poll(self):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    child=Child();monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:child);monkeypatch.setattr(m.controls.prior.shared.old.cost,'cleanup',lambda c:cleaned.append(c)or-15)
    result=m.run('checkpoint','fake');assert result['terminal']and not result['completed']and result['failure']['type']=='InterruptedError'and cleaned==[child]
    assert all(signal.getsignal(s)==h for s,h in before.items())


def test_real_guards_all_declared_cores_no_models():
    for role,slot,cpu in[('controller',None,18)]+[('worker',s,c)for s,c in enumerate(m.LIMITS['worker_CPUs'])]:
        command=['taskset','-c',str(cpu),sys.executable,'-m','experiments.run_fixture_balanced_nh_catalog_v01','guard','--role',role]
        if slot is not None:command+=['--slot',str(slot)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');v=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30);assert v.returncode==0,v.stderr


def test_no_physics_until_source_release_no_truth_rng_imports():
    assert inspect.getsource(m.group_worker).index('maximum_iterations=40')>=0
    assert 'np.random'not in Path(m.ROOT/m.SOURCE).read_text()
    assert m.LIMITS['group_wall_s']==5400 and m.LIMITS['global_wall_s']==21600 and m.LIMITS['aggregate_RSS_bytes']==76*1024**3


def test_completed_negative_first_group_retained_and_blocks_expansion(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='negative retained'):fake_terminal(monkeypatch,tmp_path,first_reference_valid=False)
    group=json.loads((tmp_path/'groups/group-00/result-LOCAL.json').read_text())
    assert group['completed']and group['all33_retained']and len(group['rows'])==33 and not group['all11_reference_valid']


def test_downstream_gate_requires_prepinned_independent_audit_no_array_fallback(monkeypatch,tmp_path):
    p,digest,_=fake_terminal(monkeypatch,tmp_path);summary=m.finalize(digest)
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No array in completedgate'))
    audit=dict(audit_passed=True,all264_retained=True,all88_reference_valid=True,source_SHA256=p['source_SHA256'],source_summary=summary,audit_source=m.record(tmp_path/m.AUDIT_SOURCE),audit_test=m.record(tmp_path/m.AUDIT_TEST))
    m.write(tmp_path/'independent-review.json',audit);value=m.completed_catalog_gate();assert len(value['groups'])==8 and value['all88_reference_valid']and value['new_fields']==264
    audit['audit_source']['sha256']='tamper';m.write(tmp_path/'independent-review.json',audit)
    with pytest.raises(ValueError,match='auditprocedure'):m.completed_catalog_gate()
