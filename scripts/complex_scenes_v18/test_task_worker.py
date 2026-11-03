import json,sys
import pytest
from scripts.complex_scenes_v18.task_worker import run,alive,key


def test_worker_records_failure_and_never_repeats_claimed_task(tmp_path):
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(tasks=[dict(id='toy-failure',argv=[sys.executable,'-c','raise SystemExit(7)'])])))
    state=tmp_path/'state';result=run(plan,'toy-failure',state)
    assert result['state']=='failed' and result['exit_code']==7
    assert not alive(result['child_pid'],result['child_start_ticks'])
    assert json.loads((state/(key('toy-failure')+'.json')).read_text())['exit_code']==7
    with pytest.raises(FileExistsError):run(plan,'toy-failure',state)
