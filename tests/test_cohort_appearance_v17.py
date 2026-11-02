import pytest
from scripts.compositional_dynamics_v17 import cohort_appearance as r
from scripts.compositional_dynamics_v17 import cohort_appearance_score as s

def test_incomplete_predictions_block_before_references(monkeypatch):
    monkeypatch.setattr(r,'prediction_gate',lambda:(_ for _ in ()).throw(ValueError('not committed')))
    monkeypatch.setattr(r,'read',lambda path:pytest.fail('Reference read before gate'))
    with pytest.raises(ValueError,match='not committed'):s.score()

def test_exact_failed_census_retained_and_duplicate_rejected(monkeypatch):
    queries=[dict(world_id=f'world-{w}',role=role,cameras=[dict(name=f'cam-{c}')for c in range(5)])for w in range(4)for role in ['forecast-40mm','forecast-80mm']]
    p=dict(queries=queries);rows=[dict(world_id=q['world_id'],query=q['role'],method=m,frame=i,time_s=i*.01,camera=c['name'],completed=False)for q in queries for m in ['native','shared_q2']for i in range(31)for c in q['cameras']]
    commit=dict(plan_sha256='same',rows=rows)
    monkeypatch.setattr(r,'guard',lambda:p);monkeypatch.setattr(r,'sha',lambda path:'same');monkeypatch.setattr(r,'read',lambda path:commit)
    assert len(r.prediction_gate()[1]['rows'])==2480
    rows[-1]=dict(rows[0])
    with pytest.raises(ValueError,match='unique'):r.prediction_gate()
