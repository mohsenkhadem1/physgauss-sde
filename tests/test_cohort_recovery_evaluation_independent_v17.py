"""Recovery gate tests without any actual prediction or target access."""
import json
import pytest
from scripts.compositional_dynamics_v17_reporting import cohort_recovery_evaluation as m

@pytest.mark.parametrize('mode',['missing','duplicate'])
def test_recovery_terminal_census_before_targets(tmp_path,monkeypatch,mode):
 monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m.recovery,'OUT',tmp_path/'recovery');monkeypatch.setattr(m.recovery,'PLAN',tmp_path/'plan.json')
 m.recovery.PLAN.write_text('{}');m.recovery.OUT.mkdir()
 monkeypatch.setattr(m.frozen,'committed',lambda b:({},[],{}));monkeypatch.setattr(m.recovery,'guard',lambda:dict(worlds=[]))
 if mode=='duplicate':
  rows=[dict(world_id='world-0',role='forecast-40mm')]*8
  (m.recovery.OUT/'recovery-manifest.json').write_text(json.dumps(dict(terminal=True,plan_sha256=m.sha(m.recovery.PLAN),expected=8,rows=rows)))
 with pytest.raises(FileNotFoundError if mode=='missing'else ValueError,match='recovery-manifest'if mode=='missing'else'census'):
  m.committed({})
