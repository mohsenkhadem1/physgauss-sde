from pathlib import Path
import json
import pytest
from scripts.phystwin import supervise_qualified_campaign as campaign


def test_failed_reference_prevents_inference_and_reports_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'STATUS',tmp_path)
    monkeypatch.setattr(campaign.sys,'argv',['supervisor'])
    stages=[]
    def reference(split,mesh=4):
        stages.append(split)
        raise RuntimeError('reference not qualified')
    monkeypatch.setattr(campaign,'reference',reference)
    monkeypatch.setattr(campaign,'policy',lambda stage:pytest.fail('Inference must not run'))
    with pytest.raises(RuntimeError,match='reference not qualified'):campaign.main()
    assert stages==['development']
    assert json.loads((tmp_path/'supervisor.json').read_text())['stage']=='stopped-on-error'


def test_worker_failure_waits_for_other_workers_before_stopping():
    seen=[]
    class Process:
        def __init__(self,code):self.code=code
        def wait(self):seen.append(self.code);return self.code
    with pytest.raises(RuntimeError,match='Phase failed'):campaign.wait_all([Process(1),Process(0),Process(0)])
    assert seen==[1,0,0]
