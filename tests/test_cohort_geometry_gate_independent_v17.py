"""Independent pre-target gate fixtures; no cohort data."""
import json
import pytest
from scripts.compositional_dynamics_v17 import cohort_geometry_evaluation as m

@pytest.mark.parametrize('foreign',[False,True])
def test_missing_or_foreign_commit_prevents_reference_access(tmp_path,monkeypatch,foreign):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'REPORT',tmp_path/'reports')
    monkeypatch.setattr(m,'guard',lambda **kwargs:dict(generator_settings=dict(world_ids=[f'world-{i}'for i in range(4)])))
    if foreign:
        folder=tmp_path/'runs/compositional-dynamics-v17/fresh-cohort/shared-forecast/world-0/forecast-40mm';folder.mkdir(parents=True)
        (folder/'prediction-commit.json').write_text(json.dumps(dict(world_id='wrong',role='forecast-40mm',protocol_sha256='p',method_freeze_sha256='f',completed=False)))
    with pytest.raises(ValueError if foreign else FileNotFoundError,match='binding'if foreign else'prediction-commit'):
        m.main(dict(protocol_sha='p',freeze_sha='f'))
    assert not (m.REPORT/'fresh-cohort-geometry-results.json').exists()
