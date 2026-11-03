import json,subprocess,sys
from scripts.complex_scenes_v18.supervise_held import observe,ready
from scripts.complex_scenes_v18.task_worker import key,start_ticks


def test_live_orphan_child_is_adopted_and_never_requeued(tmp_path):
    task=dict(id='test',dependencies=[]);child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        record=dict(task_id='test',state='running',plan_sha256='hash',pid=-1,start_ticks='absent',child_pid=child.pid,child_start_ticks=start_ticks(child.pid))
        (tmp_path/(key('test')+'.json')).write_text(json.dumps(record))
        state=observe(task,tmp_path,'hash');assert state['state']=='running'
        assert ready([task],{'test':state})==[]
    finally:child.terminate();child.wait()
    assert observe(task,tmp_path,'hash')['state']=='interrupted'
    assert ready([task],{'test':observe(task,tmp_path,'hash')})==[]


def test_abandoned_claim_is_terminal_and_dependencies_require_success(tmp_path):
    task=dict(id='claimed',dependencies=[]);(tmp_path/(key('claimed')+'.claim')).write_text('-1')
    assert observe(task,tmp_path,'hash')['state']=='interrupted'
    tasks=[dict(id='child',dependencies=['parent'])]
    assert ready(tasks,dict(child=dict(state='pending'),parent=dict(state='failed')))==[]


def test_real_worker_chain_runs_once_and_propagates_failure(tmp_path,monkeypatch):
    from scripts.complex_scenes_v18 import supervise_held
    from scripts.complex_scenes_v18.held_manifest import sha
    campaign=tmp_path/'campaign';campaign.mkdir();marker=tmp_path/'marker'
    source=tmp_path/'source';source.write_text('toy input')
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(dict(status='frozen',held_data_generated=False,protocol={},source_sha256={str(source):sha(source)},artifact_sha256={str(source):sha(source)})))
    public=campaign/'public-manifest.json';public.write_text('{}')
    (campaign/'manifest-seal.json').write_text(json.dumps(dict(freeze_sha256=sha(freeze),public_manifest_sha256=sha(public))))
    tasks=[dict(id='first',kind='cpu',role='prefix',argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).write_text("once")'],dependencies=[]),dict(id='second',kind='cpu',role='fit',argv=[sys.executable,'-c','raise SystemExit(9)'],dependencies=['first']),dict(id='third',kind='cpu',role='fit',argv=[sys.executable,'-c','raise RuntimeError("must not run")'],dependencies=['second'])]
    monkeypatch.setattr(supervise_held,'build',lambda *args:tasks)
    counts=supervise_held.run(campaign,freeze,workers=1,gpus=(0,))
    assert counts=={'complete':1,'failed':1,'dependency_failed':1}
    before=marker.stat().st_mtime_ns
    assert supervise_held.run(campaign,freeze,workers=1,gpus=(0,))==counts
    assert marker.stat().st_mtime_ns==before
