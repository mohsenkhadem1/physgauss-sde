"""Independent guard regression: foreign/missing commits fail before truth access."""
import json
import pytest
from scripts.compositional_dynamics_v17 import matched_forecast_evaluation as m

def test_foreign_coarse_plan_rejected_before_any_target_read(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'REPORT',tmp_path/'reports');monkeypatch.setattr(m,'TRUTH',tmp_path/'forbidden-truth');monkeypatch.setattr(m,'OUT',tmp_path/'result.json')
    m.REPORT.mkdir();(m.REPORT/'matched-material-forecast-plan.json').write_text('{}')
    folder=tmp_path/'runs/compositional-dynamics-v17/matched-material-forecast';folder.mkdir(parents=True)
    (folder/'prediction-commit.json').write_text(json.dumps(dict(completed=False,plan_sha256='foreign')))
    with pytest.raises(ValueError,match='Foreign forecast plan'):m.main()
    assert not m.OUT.exists()

def test_missing_commit_blocks_before_target_read(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'OUT',tmp_path/'result.json');monkeypatch.setattr(m,'TRUTH',tmp_path/'forbidden-truth')
    with pytest.raises(FileNotFoundError,match='prediction-commit'):m.main()
