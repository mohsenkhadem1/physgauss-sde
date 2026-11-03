import hashlib,json
import numpy as np
import pytest
from scripts.complex_scenes_v18 import material_scene_fit


def test_continuation_rejects_another_observation_prefix_before_simulation(tmp_path,monkeypatch):
    prefix=tmp_path/'prefix';prefix.mkdir();np.savez(prefix/'observations.npz',positions_m=np.zeros((2,1,3,3)),surface_indices=np.arange(3),times_s=[0,.002])
    meta=dict(case='clamp_release',variant=0,observations_sha256=hashlib.sha256((prefix/'observations.npz').read_bytes()).hexdigest(),calibration_duration_s=.002)
    (prefix/'metadata.json').write_text(json.dumps(meta));prior=tmp_path/'prior';prior.mkdir()
    (prior/'fit-seal.json').write_text(json.dumps(dict(completed=True,forecast_targets_read=False,observations_sha256='different-world',law_type='learned')))
    monkeypatch.setattr(material_scene_fit,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='same calibration'):material_scene_fit.fit('clamp_release',0,prefix=prefix,initialize_fit=prior)
    assert not (tmp_path/'runs').exists()
