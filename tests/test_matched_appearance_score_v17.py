import pytest
from scripts.compositional_dynamics_v17 import matched_appearance_score as s

def test_missing_prediction_blocks_private_reference_access(monkeypatch):
    p=dict(methods=['native','shared_q2'],cameras=[dict(name=str(i))for i in range(5)],times_s=[i*.01 for i in range(26)])
    monkeypatch.setattr(s.r,'guard',lambda:p);monkeypatch.setattr(s.r,'sha',lambda path:'sha')
    accessed=[]
    def read(path):
        if str(path).endswith('prediction-commit.json'):return dict(plan_sha256='sha',records=[])
        accessed.append(path);raise AssertionError('Private reference opened')
    monkeypatch.setattr(s.r,'read',read)
    with pytest.raises(ValueError,match='260'):s.score()
    assert not accessed
