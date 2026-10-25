"""Manufactured operator tests. No physical/noisy stages or campaign RNG run."""
import copy,json,os,sys,time
from pathlib import Path
import pytest
from experiments import supervise_noisy_poke_pipeline_v01 as o

@pytest.fixture
def local(monkeypatch,tmp_path):
    original=o.ROOT;monkeypatch.setenv('PYTHONPATH',str(original));monkeypatch.setattr(o,'ROOT',tmp_path);monkeypatch.setattr(o,'LOCAL',tmp_path/'operator');monkeypatch.setattr(o,'guard',lambda:None);monkeypatch.setattr(o,'preapproval',lambda digest:{})
    monkeypatch.setattr(o,'sources',lambda:{'toy':'fixed'});o.exclusive(o.LOCAL/'root-preapproval.json',dict(approved=True))
    monkeypatch.setitem(o.CAPS,'toy',[19,15,1024**3]);return tmp_path

def cmd(code):return [sys.executable,'-c',code]

def test_real_harmless_child_handshake_reap_completed_verify_and_no_retry(local):
    output=local/'hello.json';command=cmd(f'from pathlib import Path; Path({str(output)!r}).write_text("{{}}")')
    value=o.stage('toy',command,[output],'toy',parent_sha='parent')
    assert value['completed']and value['reaped']and value['exit_code']==0 and value['child_identity']['PGID']==value['child_identity']['pid']
    stamp=(o.LOCAL/'stages/toy/execution.json').stat().st_mtime_ns
    assert o.stage('toy',command,[output],'toy',parent_sha='parent')==value
    assert(o.LOCAL/'stages/toy/execution.json').stat().st_mtime_ns==stamp
    output.write_text('{"changed":true}')
    with pytest.raises(ValueError):o.stage('toy',command,[output],'toy',parent_sha='parent')

def test_partial_stage_existing_output_or_changed_membership_never_launch(local,monkeypatch):
    output=local/'result.json';output.write_text('{}');monkeypatch.setattr(o.subprocess,'Popen',lambda *a,**k:pytest.fail('No unbound replay'))
    with pytest.raises(FileExistsError):o.stage('toy',cmd('pass'),[output],'toy',parent_sha='parent')
    output.unlink();o.exclusive(o.LOCAL/'stages/toy/execution.json',dict(completed=False))
    with pytest.raises(ValueError):o.stage('toy',cmd('pass'),[output],'toy',parent_sha='parent')

def test_completed_receipt_rejects_changed_expected_outputs_parent_resources_or_source(local):
    p=local/'a';p.write_text('a');child=dict(pid=987654,starttime=1);value=dict(completed=True,reaped=True,exit_code=0,cleanup_exit_code=0,phase='toy',command=['toy'],operator_preapproval=o.record(o.LOCAL/'root-preapproval.json'),source_SHA256=o.sources(),outputs=[o.record(p)],parent_protocol_SHA256='parent',resources=dict(CPU=19,wall_s=15,RSS_bytes=1024**3),result=o.record(p),observed_owned=[child],child_identity=child)
    assert o.verify_stage(value,'toy',['toy'],[p],'toy','parent')==value
    for key,wrong in [('parent_protocol_SHA256','other'),('resources',{}),('source_SHA256',{}),('outputs',[])]:
        bad=copy.deepcopy(value);bad[key]=wrong
        with pytest.raises(ValueError):o.verify_stage(bad,'toy',['toy'],[p],'toy','parent')
    q=local/'new';q.write_text('new')
    with pytest.raises(ValueError):o.verify_stage(value,'toy',['toy'],[p,q],'toy','parent')

def test_real_owned_timeout_cleans_only_harmless_child_and_retains_negative(local,monkeypatch):
    monkeypatch.setitem(o.CAPS,'toy',[19,.2,1024**3]);calls=[];real=o.os.killpg
    monkeypatch.setattr(o.os,'killpg',lambda pid,sig:calls.append((pid,sig))or real(pid,sig))
    with pytest.raises(TimeoutError):o.stage('toy',cmd('import time; time.sleep(20)'),[],'toy',parent_sha='parent')
    value=o.read(o.LOCAL/'stages/toy/execution.json');assert not value['completed']and value['reaped'];assert all(pid==value['child_identity']['pid']for pid,sig in calls)
    with pytest.raises(ValueError):o.stage('toy',cmd('pass'),[],'toy',parent_sha='parent')

def test_unknown_child_identity_and_reused_pid_never_signalled(local,monkeypatch):
    class Child:
        pid=987654
        def poll(self):return None
    monkeypatch.setattr(o.os,'killpg',lambda *a:pytest.fail('No unknown kill'))
    with pytest.raises(RuntimeError,match='No proven'):o.cleanup(Child(),[])
    monkeypatch.setattr(o,'identity',lambda pid:dict(pid=pid,starttime=999,state='R'))
    assert not o.alive(dict(pid=987654,starttime=1))

def checkpoint_fixture(local,monkeypatch,qualified=True,wall=1.):
    monkeypatch.setattr(o.noisy,'LOCAL',local/'noisy');entries=[]
    for i in range(8):
        path=local/f'post-{i}.json';o.exclusive(path,dict(wall_s=wall));entries.append(dict(qualified=True,record=o.record(path)))
    path=local/'trial.json';o.exclusive(path,dict(qualified=qualified,posteriors=entries));worker=local/'worker.json';o.exclusive(worker,dict(completed=True,trials=[dict(index=0,record=o.record(path))]));monkeypatch.setattr(o.noisy,'completed_phase',lambda *a:dict(workers=[dict(result=o.record(worker))]))

def test_fixed_checkpoint_negative_or_excesscost_cannot_generate_release(local,monkeypatch):
    checkpoint_fixture(local,monkeypatch,qualified=False)
    with pytest.raises(ValueError,match='qualification'):o.checkpoint_admission('toy')
    assert not(o.noisy.LOCAL/'continuation-release.json').exists()
    path=local/'trial.json';value=o.read(path);value['qualified']=True;o.atomic(path,value)
    # The helper intentionally reads only original records; fix worker binding for toy.
    assert o.checkpoint_admission('toy')==264
    for i in range(8):o.atomic(local/f'post-{i}.json',dict(wall_s=1000.))
    with pytest.raises(ValueError,match='budget'):o.checkpoint_admission('toy')
    assert not(o.noisy.LOCAL/'continuation-release.json').exists()

def test_commit_is_exclusive_and_stable_not_overwrite(local):
    path=local/'release.json';binding=o.commit(path,dict(approved=True));assert o.commit(path,dict(approved=True))==binding
    with pytest.raises(ValueError):o.commit(path,dict(approved=False))

def test_source_only_pipeline_all88failure_stops_before_noisy_prepare(local,monkeypatch):
    audit=local/'audit.json';o.exclusive(audit,dict(audit_passed=True,all33_retained=True,all11_reference_valid=True));p=dict(checkpoint_independent_review=o.record(audit));monkeypatch.setattr(o,'preapproval',lambda _:p)
    monkeypatch.setattr(o.physics,'validate',lambda _:dict(source_SHA256={}));monkeypatch.setattr(o.physics,'continuation_gate',lambda *a:None);seen=[]
    monkeypatch.setattr(o,'stage',lambda name,*a,**k:seen.append(name));monkeypatch.setattr(o,'controller_closure',lambda *a:None)
    monkeypatch.setattr(o.physics,'completed_catalog_gate',lambda:(_ for _ in()).throw(ValueError('Only87valid')))
    with pytest.raises(ValueError,match='87'):o.pipeline('toy')
    assert seen==['physics_remaining','physics_finalize','physics_audit']

def test_preapproval_exactschema_source_policy_and_existing_roots_only(local,monkeypatch):
    # Exercise the unpatched parser against manufactured root metadata only.
    import importlib
    real=importlib.reload(o).preapproval
    monkeypatch.setattr(o,'ROOT',local);monkeypatch.setattr(o,'LOCAL',local/'operator');monkeypatch.setattr(o,'sources',lambda:{'toy':'pin'});monkeypatch.setattr(o.physics,'LOCAL',local/'physics')
    records={}
    for key,name in [('physics_release','release.json'),('physics_remaining_release','continuation-release.json'),('checkpoint_independent_review','checkpoint-independent-review.json')]:
        path=o.physics.LOCAL/name;path.parent.mkdir(exist_ok=True);path.write_text('{}');records[key]=o.record(path)
    p=dict(approved=True,schema='noisy-poke-operator-preapproval-v01',source_SHA256={'toy':'pin'},physics_protocol_SHA256=o.PHYSICS_SHA,prospective_design_SHA256=o.DESIGN_SHA,stage_caps=o.CAPS,operator_resources=o.OPERATOR_LIMITS,dynamic_release_policy=o.POLICY,**records);o.atomic(o.LOCAL/'root-preapproval.json',p)
    assert real(o.sha(o.LOCAL/'root-preapproval.json'))==p
    p['dynamic_release_policy']={};o.atomic(o.LOCAL/'root-preapproval.json',p)
    with pytest.raises(ValueError):real(o.sha(o.LOCAL/'root-preapproval.json'))


@pytest.mark.parametrize('final_audit_ok',[True,False])
def test_manufactured_whole_pipeline_orders_gates_and_dynamic_releases(local,monkeypatch,final_audit_ok):
    monkeypatch.delitem(o.CAPS,'toy')
    for module,name in((o.physics,'physics'),(o.noisy,'noisy'),(o.finite,'finite')):monkeypatch.setattr(module,'LOCAL',local/name)
    review=local/'checkpoint.json';o.exclusive(review,dict(audit_passed=True,all33_retained=True,all11_reference_valid=True));monkeypatch.setattr(o,'preapproval',lambda _:dict(checkpoint_independent_review=o.record(review)))
    monkeypatch.setattr(o.physics,'validate',lambda _:dict(source_SHA256={}));monkeypatch.setattr(o.physics,'continuation_gate',lambda *a:None)
    monkeypatch.setattr(o.physics,'completed_catalog_gate',lambda:events.append('ALL88VALID'))
    monkeypatch.setattr(o.noisy,'sources',lambda:{'noisy':'pin'});monkeypatch.setattr(o.noisy,'design_gate',lambda:{'design':'pin'});monkeypatch.setattr(o.finite,'sources',lambda:{'finite':'pin'})
    events=[];seals=[dict(index=i)for i in range(32)]
    def noop_validate(ns):assert(o.noisy.LOCAL/'release.json').is_file();events.append('NOISYRELEASE');return {}
    monkeypatch.setattr(o.noisy,'validate',noop_validate);monkeypatch.setattr(o.noisy,'all_seals_gate',lambda _:events.append('ALL32SEALED')or seals)
    def admitted(ns):assert not(o.noisy.LOCAL/'continuation-release.json').exists();events.append('COSTQUALIFIED');return 1.
    monkeypatch.setattr(o,'checkpoint_admission',admitted)
    def continuation(ns):assert(o.noisy.LOCAL/'continuation-release.json').is_file();events.append('CONTINUATIONRELEASE')
    monkeypatch.setattr(o.noisy,'continuation_gate',continuation);monkeypatch.setattr(o,'controller_closure',lambda *a:None);monkeypatch.setattr(o,'finite_closure',lambda *a:None)
    def staged(name,cmd,outputs,digest,parent_sha):
        events.append(name)
        if name=='noisy_prepare':assert'ALL88VALID'in events
        if name=='finite_choose':assert'ALL32SEALED'in events
        if name=='noisy_remaining':assert'CONTINUATIONRELEASE'in events
        for path in outputs:
            value=dict(audit_passed=final_audit_ok if name=='finite_audit'else True)
            if name=='noisy_prepare':value=dict(source_SHA256=o.noisy.sources(),resources=o.noisy.RESOURCES,origin=dict(prospective_design=o.noisy.design_gate()))
            o.exclusive(path,value)
        o.exclusive(o.LOCAL/'stages'/name/'execution.json',dict(completed=True));return {}
    monkeypatch.setattr(o,'stage',staged)
    if not final_audit_ok:
        with pytest.raises(ValueError,match='finite integrity'):o.pipeline('toy')
        assert not(o.LOCAL/'completed.json').exists();return
    o.pipeline('toy')
    names=[v for v in events if v in o.CAPS];assert names==list(o.CAPS)
    assert events.index('COSTQUALIFIED')<events.index('CONTINUATIONRELEASE')<events.index('noisy_remaining')
    complete=o.read(o.LOCAL/'completed.json');assert complete['completed']and len(complete['stages'])==13


def test_full_scientific_operator_and_reserve_preflight_before_spawn(local,monkeypatch):
    monkeypatch.setattr(o.noisy,'available_memory',lambda:o.CAPS['physics_remaining'][2]+30*o.GiB)
    monkeypatch.setattr(o.subprocess,'Popen',lambda *a,**k:pytest.fail('No spawn below110GiB envelope'))
    with pytest.raises(MemoryError,match='envelope'):o.stage('physics_remaining',['toy'],[],'toy',parent_sha='parent')
    assert not(o.LOCAL/'stages/physics_remaining/execution.json').exists()


def test_descendant_scan_refuses_reused_direct_pid(local,monkeypatch):
    monkeypatch.setattr(o,'identity',lambda pid:dict(pid=pid,starttime=999))
    with pytest.raises(RuntimeError,match='reused'):o.descendants(dict(pid=987654,starttime=1))

def test_uncooperative_proven_owned_sessions_receive_65second_TERM_grace(local,monkeypatch):
    clock=[0.];living={987651,987652};calls=[]
    rows=[dict(pid=p,PGID=p,starttime=1)for p in living]
    class Child:
        pid=987651
        def poll(self):return None
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(o,'alive',lambda row:row['pid']in living)
    monkeypatch.setattr(o,'descendants',lambda row:rows)
    monkeypatch.setattr(o.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(o.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    def signal(pid,sig):
        calls.append((pid,sig,clock[0]))
        if sig==o.signal.SIGKILL:living.remove(pid)
    monkeypatch.setattr(o.os,'killpg',signal);o.cleanup(Child(),rows)
    assert not living and{pid for pid,sig,t in calls}=={987651,987652}
    assert all(t>=65 for pid,sig,t in calls if sig==o.signal.SIGKILL)


def test_dead_group_leader_live_observed_nonleader_is_signalled_individually(local,monkeypatch):
    living={987652};calls=[];rows=[dict(pid=987651,PGID=987651,starttime=1),dict(pid=987652,PGID=987651,starttime=2)]
    class Child:
        pid=987651
        def poll(self):return 0
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(o,'alive',lambda row:row['pid']in living)
    monkeypatch.setattr(o.os,'killpg',lambda *a:pytest.fail('Never signal dead/unknown group leader'))
    def signal(pid,sig):calls.append((pid,sig));living.remove(pid)
    monkeypatch.setattr(o.os,'kill',signal);o.cleanup(Child(),rows)
    assert calls==[(987652,o.signal.SIGTERM)]and not living
