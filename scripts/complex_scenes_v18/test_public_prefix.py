import hashlib,json
import numpy as np
import pytest
from scripts.complex_scenes_v18.export_public_prefix import export


def test_export_removes_future_and_private_parameters(tmp_path):
    source=tmp_path/'reference';source.mkdir();q=np.array([[0,0,0],[1,1,1],[.5,.5,0]])
    values=np.arange(4*1*3*3).reshape(4,1,3,3)
    raw=source/'trajectory.npz';np.savez(raw,times_s=np.arange(4)*.002,material_queries=q,surface_positions_m=values)
    status=dict(completed=True,trajectory_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),public_query=dict(family='clamp_release',variant=0),world_id='private-world',private_world_sha256='commitment',private_material_values=[12345.])
    (source/'complete.json').write_text(json.dumps(status));out=tmp_path/'prefix';meta=export(source,out,.002)
    with np.load(out/'observations.npz')as a:
        assert a['positions_m'].shape==(2,1,2,3)
        np.testing.assert_array_equal(a['positions_m'],values[:2,:,[0,1]])
    assert 'private_material_values'not in meta and not meta['forecast_targets_exported']
    with pytest.raises(ValueError):export(source,tmp_path/'bad',.006)
