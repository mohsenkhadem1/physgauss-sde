import hashlib
import numpy as np
from scripts.complex_scenes_v18 import world_queries,run_development


def test_public_world_run_saves_auditable_full_state(tmp_path,monkeypatch):
    original=world_queries.build_reference
    def short(*args,**kwargs):
        scene,state,meta=original(*args,**kwargs);meta['duration_s']=.002
        return scene,state,meta
    monkeypatch.setattr(world_queries,'build_reference',short);monkeypatch.setattr(run_development,'ROOT',tmp_path)
    world=world_queries.World('export-check',(world_queries.Material(float(np.log(12000))),))
    query=world_queries.Query('container_drop',0,initial_height_m=.02)
    result=run_development.run('container_drop',1,50e-6,world=world,query=query)
    assert result['completed']and result['full_nodal_state_saved']
    folder=tmp_path/result['path'];path=folder/'nodal-state.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==result['nodal_state_sha256']
    with np.load(path)as a:
        assert a['positions_m'].shape==(2,1,81,3)
        np.testing.assert_allclose(a['velocities_m_s'][-1,0],np.tile([0,0,-9.81*.002],(81,1)),atol=1e-9)
    from scripts.complex_scenes_v18.impulse_metrics import trace
    _,impulse=trace(folder)
    np.testing.assert_allclose(impulse,0.,atol=1e-10)
    from scripts.complex_scenes_v18.appearance_controls import controls
    times,positions=controls(folder)
    assert positions.shape==(2,1,125,3)
    np.testing.assert_allclose(positions[-1,0]-positions[0,0],np.tile([0,0,-.5*9.81*.002**2],(125,1)),atol=1e-10)
