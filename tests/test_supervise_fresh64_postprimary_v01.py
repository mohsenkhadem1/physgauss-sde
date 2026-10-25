"""Manufactured metadata/processes only; no campaign calls or arrays."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
from experiments import supervise_fresh64_postprimary_v01 as m

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def terminal():return dict(controller_pid=m.PARENT_PID,terminal=True,completed=True,workers=[dict(pid=p,reaped=True,exit_code=0)for p in m.PARENT_WORKERS])
def test_parent_wait_only_checks_identity_closure_and_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(m.continued.parent,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None);monkeypatch.setattr(m,'identity',lambda _:None);seen=[];monkeypatch.setattr(m,'absent',seen.append)
    write(tmp_path/'execution.json',terminal());assert m.parent_terminal(dict(parent_identity=None));assert seen==[m.PARENT_PID,*m.PARENT_WORKERS]
    value=terminal();value['workers'][0]['exit_code']=1;write(tmp_path/'execution.json',value)
    with pytest.raises(ValueError,match='execution failed'):m.parent_terminal(dict(parent_identity=None))
    value=terminal();value['controller_pid']+=1;write(tmp_path/'execution.json',value)
    with pytest.raises(ValueError,match='original parent'):m.parent_terminal(dict(parent_identity=None))
def test_PID_reuse_refused_without_signals(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'check_sources',lambda _:None);monkeypatch.setattr(m,'identity',lambda _:dict(starttime=2));monkeypatch.setattr(m.os,'killpg',lambda *a:pytest.fail('Parent must never be signaled'))
    with pytest.raises(ValueError,match='PID reuse'):m.parent_terminal(dict(parent_identity=dict(starttime=1)))
def test_original_audit_numeric_block_required():
    v=dict(completed=True,integrity_passed=True,execution_clean=True,comparison_blocked=True,objects=64,qualified_posteriors=889);m.original_audit_admission(v)
    for key in('integrity_passed','execution_clean','comparison_blocked'):
        with pytest.raises(ValueError):m.original_audit_admission(dict(v,**{key:False}))
def test_census_no_filter_both_axis_block_and_cost_bound():
    fine=dict(cases={'toy':dict(seconds=100.)});job=dict(decision='refine',plan=dict(inner=[128,256,512,1024],outer=[128,256,512]));c=dict(counts=dict(blocked=0),entries=[job]*9)
    assert m.continuation_admission(c,fine)==9
    with pytest.raises(ValueError,match='cost admission'):m.continuation_admission(c,dict(cases={'toy':dict(seconds=4000.)}))
    with pytest.raises(ValueError,match='Blocked'):m.continuation_admission(dict(c,counts=dict(blocked=1)),fine)
    both=dict(job,plan=dict(inner=[128,256,512,1024],outer=[128,256,512,1024]))
    with pytest.raises(ValueError,match='Both-axis'):m.continuation_admission(dict(c,entries=[both]),fine)
def test_stage_resume_verified_no_relaunch_and_partial_never_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None);monkeypatch.setattr(m,'absent',lambda _:None)
    output=tmp_path/'artifact.json';write(output,{'ok':True});path=tmp_path/'stages/toy/execution.json';cmd=['no-call'];write(tmp_path/'protocol.json',{})
    write(path,dict(completed=True,command=cmd,protocol_SHA256=m.sha(tmp_path/'protocol.json'),outputs=[m.record(output)],child=dict(pid=123)))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('No completed-stage replay'))
    assert m.stage('toy',cmd,[output],{})['completed']
    output.write_text('tampered')
    with pytest.raises(ValueError,match='output membership/bytes'):m.stage('toy',cmd,[output],{})
    write(path,dict(completed=False))
    with pytest.raises(RuntimeError,match='no automatic retry'):m.stage('toy',cmd,[output],{})
def test_no_preexisting_output_or_uncommitted_stage_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None);output=tmp_path/'old.json';output.write_text('{}')
    with pytest.raises(FileExistsError):m.stage('toy',['none'],[output],{})
def test_cleanup_refuses_parent_even_if_claimed_owned(monkeypatch):
    child=object()
    with pytest.raises(ValueError,match='Refuse parent'):m.cleanup(child,[dict(pid=m.PARENT_PID)])
def test_owned_child_identity_and_cleanup_does_not_signal_unrelated():
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        owned=m.identity(child.pid);assert owned['PGID']==child.pid
        m.cleanup(child,[owned]);assert child.returncode is not None and other.poll()is None
    finally:
        if child.poll()is None:os.killpg(child.pid,9);child.wait()
        os.killpg(other.pid,15);other.wait()
def test_missing_live_RSS_fails_closed_after_bounded_grace(monkeypatch):
    monkeypatch.setattr(m,'alive',lambda _:True);monkeypatch.setattr(Path,'read_text',lambda *a,**k:'State:\tR\n')
    with pytest.raises(RuntimeError,match='RSS unresolved'):m.owned_RSS(dict(pid=999))
def test_short_lived_real_RSS_exit_races():
    for _ in range(20):
        child=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        try:
            row=m.identity(child.pid)
            while child.poll()is None:
                if row:assert m.owned_RSS(row)>=0
            child.wait()
            if row:assert m.owned_RSS(row)==0
        finally:
            if child.poll()is None:child.kill();child.wait()
def test_root_authorization_before_parent_wait(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None);write(tmp_path/'protocol.json',{})
    write(tmp_path/'release.json',dict(approved=False))
    with pytest.raises(ValueError,match='Explicit root'):m.supervisor_release(m.sha(tmp_path/'protocol.json'))
def test_fixed_resource_scope_and_caps():
    r=m.resources();assert r['workers']==8 and r['aggregate_RSS_bytes']==36*m.GiB and not r['GPU']and r['global_wall_s']==19800


def test_parent_terminal_write_to_exit_gap_waits_and_corruption_only_alive(tmp_path,monkeypatch):
    monkeypatch.setattr(m.continued.parent,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None)
    now=dict(starttime=123,state='R');monkeypatch.setattr(m,'identity',lambda _:now);monkeypatch.setattr(m,'absent',lambda _:(_ for _ in()).throw(ValueError('still present')))
    p=dict(parent_identity=now);write(tmp_path/'execution.json',terminal());assert m.parent_terminal(p)is False
    (tmp_path/'execution.json').write_text('{partial');assert m.parent_terminal(p)is False
    monkeypatch.setattr(m,'identity',lambda _:None)
    with pytest.raises(ValueError,match='Corrupt parent'):m.parent_terminal(p)

def test_parent_worker_PID_reuse_refuses_without_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(m.continued.parent,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None);write(tmp_path/'execution.json',terminal())
    monkeypatch.setattr(m,'identity',lambda pid:dict(starttime=2)if pid==m.PARENT_WORKERS[0]else None)
    with pytest.raises(ValueError,match='worker PID reused'):m.parent_terminal(dict(parent_identity=None,parent_worker_identities={str(m.PARENT_WORKERS[0]):dict(starttime=1)}))


def test_real_harmless_stage_handshake_commit_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None)
    write(tmp_path/'protocol.json',{})
    output=tmp_path/'toy-result.json';command=m.stage_command('json',[],19)
    command=['taskset','-c','19',sys.executable,'-c',f'from pathlib import Path;Path({str(output)!r}).write_text("{{}}")']
    p=dict(stage_caps={'toy':[19,20,4*m.GiB]})
    # record uses the real ROOT, so place only its manufactured archive mapping
    # under the test workspace; no campaign artifacts are consulted.
    monkeypatch.setattr(m,'record',lambda path:dict(path=str(path),sha256=m.sha(path)))
    monkeypatch.setattr(m,'exact',lambda binding:Path(binding['path'])if m.sha(binding['path'])==binding['sha256']else pytest.fail('changed'))
    state=m.stage('toy',command,[output],p)
    assert state['completed']and state['reaped']and state['exit_code']==0
    assert m.stage('toy',command,[output],p)==state


def test_real_peer_receipts_metadata_schema_and_pins_only(monkeypatch):
    monkeypatch.setattr(m.continued.np,'load',lambda *a,**k:pytest.fail('No campaign arrays for receipt validation'))
    receipts=m.reviewed_receipts()
    assert len(receipts)==5 and all(len(v['sha256'])==64 for v in receipts)


def test_completed_stage_cannot_skip_changed_expected_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'check_sources',lambda _:None)
    write(tmp_path/'protocol.json',{});a=tmp_path/'a.json';b=tmp_path/'b.json';write(a,{});write(b,{})
    state=dict(completed=True,command=['none'],protocol_SHA256=m.sha(tmp_path/'protocol.json'),outputs=[m.record(a)])
    write(tmp_path/'stages/toy/execution.json',state)
    with pytest.raises(ValueError,match='output membership'):m.stage('toy',['none'],[a,b],{})


def test_completed_pipeline_verifies_journals_and_outputs_on_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'absent',lambda _:None)
    output=tmp_path/'final.json';write(output,{});journal=tmp_path/'stages/toy/execution.json'
    write(journal,dict(completed=True,protocol_SHA256='p',outputs=[m.record(output)]))
    write(tmp_path/'completed.json',dict(completed=True,protocol_SHA256='p',stages={'toy':m.record(journal)}))
    p=dict(stage_caps={'toy':[]});assert m.verify_completed(p,'p')['completed']
    output.write_text('changed')
    with pytest.raises(ValueError,match='Changed pinned'):m.verify_completed(p,'p')
    output.unlink()
    with pytest.raises(FileNotFoundError):m.verify_completed(p,'p')


def test_captured_parent_command_protocol_and_session_cannot_be_unrelated(monkeypatch):
    def fake(pid):return dict(pid=pid,PGID=pid,PPID=m.PARENT_PID,state='R',command='unrelated user job')
    monkeypatch.setattr(m,'identity',fake)
    with pytest.raises(ValueError,match='original command'):m.parent_identities()
