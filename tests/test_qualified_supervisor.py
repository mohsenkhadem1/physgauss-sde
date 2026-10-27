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


@pytest.mark.parametrize('complete', [True,False])
def test_resume_requires_complete_outputs_and_valid_seal(tmp_path,monkeypatch,complete):
    from scripts.phystwin import qualified_policy
    analysis=tmp_path/'analysis';analysis.mkdir()
    (analysis/'seal.json').write_text('{}')
    suffixes=['summary.json','predictions.npz','object-mse.json','utility.json','deformation-object-mse.json']
    for suffix in suffixes if complete else suffixes[:-1]:
        (analysis/f'calibration-{suffix}').touch()
    monkeypatch.setattr(campaign,'BASE',tmp_path)
    seen=[]
    monkeypatch.setattr(qualified_policy,'verify_seal',lambda seal:seen.append('verified'))
    monkeypatch.setattr(campaign,'status',lambda stage:seen.append(stage))
    monkeypatch.setattr(campaign,'launch',lambda *args:seen.append('launched'))
    monkeypatch.setattr(campaign,'wait_all',lambda processes:None)
    campaign.policy('calibrate')
    assert seen[0]=='verified'
    assert ('launched' in seen) is not complete
    def invalid(seal):raise AssertionError('Sealed file changed')
    monkeypatch.setattr(qualified_policy,'verify_seal',invalid)
    seen.clear()
    with pytest.raises(AssertionError,match='Sealed file changed'):campaign.policy('calibrate')
    assert not seen
