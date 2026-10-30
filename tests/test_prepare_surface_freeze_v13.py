import pytest
from scripts.coupled_contact_v13.prepare_surface_freeze import make_protocol


def test_candidate_protocol_exact_scope_maps_and_observation_coordinates():
    template=dict(forecast_scenarios=['slide','edge_drop','curved','moving_sphere'],observation_nodes=18)
    decision=dict(status='adopted',adopted_scope='primary_with_stress',primary_forecast_scenarios=['slide','curved','moving_sphere'],stress_forecast_scenarios=['edge_drop'])
    p=make_protocol(template,decision)
    assert p['coarse_cells']==[4]*3 and p['dt_coarse']==12.5e-6
    assert len(p['observation_node_indices'])==len(p['observation_node_coordinates_m'])==18
    assert p['reference_settings_by_scenario']['moving_sphere']==dict(cells=[12]*3,companion_cells=[10]*3,dt=25e-6)
    assert p['reference_settings_by_scenario']['edge_drop']==dict(cells=[8]*3,companion_cells=[12]*3,dt=12.5e-6)
    assert p['reference_settings_by_scenario']['squeeze']==dict(cells=[8]*3,companion_cells=[10]*3,dt=25e-6)
    assert 'coarse_cells' not in template


def test_pending_or_wrong_scope_decision_cannot_prepare():
    with pytest.raises(RuntimeError,match='Explicit matching'):
        make_protocol({},dict(status='pending'))


def test_reserved_output_guard_rejects_files_but_allows_development(tmp_path,monkeypatch):
    from scripts.coupled_contact_v13 import prepare_surface_freeze as helper
    monkeypatch.setattr(helper,'ROOT',tmp_path);monkeypatch.setattr(helper,'REPORT',tmp_path/'reports')
    helper.REPORT.mkdir()
    p=dict(objects=[dict(id='dev0',split='development'),dict(id='fresh0',split='reserved')])
    base=tmp_path/'runs/coupled-surface-v13'
    dev=base/'reference/dev0/squeeze';dev.mkdir(parents=True);(dev/'trajectory.npz').write_bytes(b'dev')
    helper.no_fresh_reserved(p)
    for directory in ['reference','reference-refinement','reference-spatial','prediction','evaluation','inference','ablation/prediction']:
        folder=base/directory/'fresh0'/'scene';folder.mkdir(parents=True,exist_ok=True)
        helper.no_fresh_reserved(p)
        file=folder/'partial.npz';file.write_bytes(b'fresh')
        with pytest.raises(RuntimeError,match='Fresh reserved artifacts'):helper.no_fresh_reserved(p)
        file.unlink()
    (helper.REPORT/'reserved-results.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Fresh reserved result'):helper.no_fresh_reserved(p)
