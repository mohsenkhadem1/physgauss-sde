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
