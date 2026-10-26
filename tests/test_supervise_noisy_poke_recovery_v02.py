"""Manufactured operator tests. No physical/noisy stages or campaign RNG run."""

import copy,json,os,sys,time

from pathlib import Path

import pytest

from experiments import supervise_noisy_poke_recovery_v02 as o

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

def test_commit_is_exclusive_and_stable_not_overwrite(local):
    path=local/'release.json';binding=o.commit(path,dict(approved=True));assert o.commit(path,dict(approved=True))==binding
    with pytest.raises(ValueError):o.commit(path,dict(approved=False))

def test_full_scientific_operator_and_reserve_preflight_before_spawn(local,monkeypatch):
    monkeypatch.setattr(o.noisy,'available_memory',lambda:o.CAPS['remaining'][2]+30*o.GiB)
    monkeypatch.setattr(o.subprocess,'Popen',lambda *a,**k:pytest.fail('No spawn below62GiB envelope'))
    with pytest.raises(MemoryError,match='envelope'):o.stage('remaining',['toy'],[],'toy',parent_sha='parent')
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

def test_thousands_of_dead_nonleaders_have_bounded_history_and_digest(monkeypatch):
    leaders=[dict(pid=800001,PGID=800001,starttime=1),dict(pid=800002,PGID=800002,starttime=2)]
    special=[dict(pid=800003,PGID=800001,starttime=3),dict(pid=800004,PGID=800001,starttime=4),dict(pid=800005,PGID=800001,starttime=5)]
    def current(pid):
        if pid==800003:return dict(pid=pid,starttime=3,state='R')
        if pid==800004:return dict(pid=pid,starttime=4,state='Z')
        if pid==800005:raise PermissionError('Unknown status must remain owned')
        return None
    monkeypatch.setattr(o,'identity',current);monkeypatch.setattr(o,'pid_directory_exists',lambda pid:False)
    kept=leaders+special;retired=o.retirement_summary();peak=0
    for batch in range(100):
        transient=[dict(pid=1000000+batch*50+i,PGID=800001,starttime=100+i)for i in range(50)]
        kept,retired=o.compact_owned(kept,transient,retired);peak=max(peak,len(kept))
    assert peak==5 and kept==leaders+special and retired['count']==5000 and retired['PID_absent']==5000 and retired['starttime_changed']==0
    assert len(json.dumps(retired))<400 and retired['SHA256_chain']!='0'*64
    again,summary=o.compact_owned(kept,[],retired);assert again==kept and summary==retired

def test_reused_pid_never_erases_old_leader_or_live_new_identity(monkeypatch):
    oldleader=dict(pid=800001,PGID=800001,starttime=1);oldhelper=dict(pid=800002,PGID=800001,starttime=2);newhelper=dict(pid=800002,PGID=800001,starttime=9)
    monkeypatch.setattr(o,'identity',lambda pid:dict(pid=pid,starttime=9,state='R'))
    kept,summary=o.compact_owned([oldleader,oldhelper],[newhelper],o.retirement_summary())
    assert kept==[oldleader,newhelper]and summary['count']==summary['starttime_changed']==1
    # Repeat input ordering has the same deterministic retirement commitment.
    _,again=o.compact_owned([oldhelper,oldleader],[newhelper],o.retirement_summary());assert summary==again

def test_none_without_confirmed_pid_absence_and_errors_are_not_retirement(monkeypatch):
    row=dict(pid=800001,PGID=800000,starttime=1);monkeypatch.setattr(o,'identity',lambda pid:None);monkeypatch.setattr(o,'pid_directory_exists',lambda pid:True)
    assert o.retirement_proof(row)is None
    def unknown(pid):raise PermissionError('Unreadable live identity')
    monkeypatch.setattr(o,'identity',unknown);assert o.retirement_proof(row)is None
    kept,summary=o.compact_owned([row],[],o.retirement_summary());assert kept==[row]and summary['count']==0

def test_live_observed_nonleader_with_dead_leader_remains_cleanup_eligible(monkeypatch):
    rows=[dict(pid=800001,PGID=800001,starttime=1),dict(pid=800002,PGID=800001,starttime=2)]
    monkeypatch.setattr(o,'identity',lambda pid:None if pid==800001 else dict(pid=pid,starttime=2,state='R'))
    monkeypatch.setattr(o,'pid_directory_exists',lambda pid:False)
    kept,summary=o.compact_owned(rows,[],o.retirement_summary());assert kept==rows and summary['count']==0
