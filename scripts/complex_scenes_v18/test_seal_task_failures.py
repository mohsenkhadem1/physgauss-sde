import json
from scripts.complex_scenes_v18 import seal_task_failures


def test_fit_failure_and_infrastructure_failure_remain_distinct(tmp_path,monkeypatch):
    monkeypatch.setattr(seal_task_failures,'ROOT',tmp_path)
    fit=dict(id='fit',role='fit',argv=['python','--label','fit-label'],dependencies=[])
    prediction=dict(id='predict',role='prediction',dependencies=['fit'])
    tasks={t['id']:t for t in [fit,prediction]};states=dict(fit=dict(state='failed'),predict=dict(state='dependency_failed'))
    assert seal_task_failures.failure_kind(prediction,tasks,states)[0]=='execution_failed'
    folder=tmp_path/'runs/complex-scenes-v18/fit-label/case';folder.mkdir(parents=True)
    (folder/'fit-seal.json').write_text(json.dumps(dict(completed=False)))
    assert seal_task_failures.failure_kind(prediction,tasks,states)[0]=='fit_failed'
    states['fit']['state']='interrupted'
    assert seal_task_failures.failure_kind(prediction,tasks,states)[0]=='execution_failed'
