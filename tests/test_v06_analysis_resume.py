import json
import pytest
from scripts.phystwin import v06_analysis_runner as runner
from scripts.phystwin import sequential_policy

def test_completed_sealed_run_is_reused_and_changed_fixed_design_rejected(tmp_path,monkeypatch):
 directory=tmp_path/'runs/phystwin-sequential-support-v06/analysis';directory.mkdir(parents=True)
 (directory/'seal.json').write_text(json.dumps(dict(fixed_sequence=[1,4],hashes={})))
 (directory/'design.json').write_text(json.dumps(dict(fixed_sequence=[1,4])))
 for stage in ['calibration','test']:(directory/f'{stage}-summary.json').write_text('{}')
 monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner.sys,'argv',['runner','--observations','direct'])
 monkeypatch.setattr(runner.subprocess,'run',lambda *args,**kwargs:pytest.fail('Completed inference must not be rerun'))
 checked=[];monkeypatch.setattr(sequential_policy,'verify',lambda seal:checked.append(True))
 runner.main();assert len(checked)==3
 (directory/'design.json').write_text(json.dumps(dict(fixed_sequence=[2,3])))
 with pytest.raises(AssertionError):runner.main()
