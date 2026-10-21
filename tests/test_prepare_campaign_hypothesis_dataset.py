"""Check that known-control features cannot inherit latent-dependent clocks."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from experiments import campaign_hypothesis_inputs as inputs
from experiments import prepare_campaign_hypothesis_dataset as dataset


@pytest.fixture(scope='module')
def manifest():
    return inputs.make_manifest(program_ids=['exciting_world_a_v01','exciting_world_b_v01'])


def test_all_material_and_boundary_counterfactuals_share_public_controls(manifest):
    public = manifest['public_hardware'][0]
    rows = [h for h in manifest['hypotheses'] if h['public_hardware_id']==public['id']]
    features = [dataset.input_features(public,h['spec'],'exciting_world_a_v01') for h in rows]
    assert len(features)==12
    for f in features:
        np.testing.assert_array_equal(f['controls'],features[0]['controls'])
        np.testing.assert_array_equal(f['controls'][0],np.zeros(12))
        assert not set(f).intersection({'id','ids','q','integration_hz','evaluation_role','train_mask'})
    assert len({f['material'][0] for f in features})==3
    assert {int(f['fixture_mask'].sum()) for f in features}=={1,16,18}
    assert all(np.all(f['support_offsets']==0) for h,f in zip(rows,features) if h['spec']['boundary_kind']!='moving_patch')


def test_hypothesis_changes_do_not_change_force_and_second_program_does(manifest):
    public = manifest['public_hardware'][0]
    h = manifest['hypotheses'][0]['spec']
    a = dataset.input_features(public,h,'exciting_world_a_v01')
    b = dataset.input_features(public,h,'exciting_world_b_v01')
    assert np.max(np.abs(a['controls']-b['controls']))>1e-3
    for key in a:
        if key!='controls':np.testing.assert_array_equal(a[key],b[key])
    assert list(inspect.signature(dataset.input_features).parameters)==['public','hypothesis','program_id']


def test_moving_support_features_equal_analytic_targets_and_full_hardware(manifest):
    h = next(h for h in manifest['hypotheses'] if h['spec']['boundary_kind']=='moving_patch')
    public = next(p for p in manifest['public_hardware'] if p['id']==h['public_hardware_id'])
    f = dataset.input_features(public,h['spec'],'exciting_world_a_v01')
    np.testing.assert_array_equal(f['support_offsets'][30],h['spec']['motion_amplitude_m']*np.asarray(h['spec']['motion_direction_world']))
    np.testing.assert_array_equal(f['support_offsets'][0],np.zeros(3))
    for j,handle in enumerate(public['handles']):
        np.testing.assert_array_equal(f['handle_inertias'][j],handle['com_inertia_body_kg_m2'])
        np.testing.assert_array_equal(f['handle_com_offsets'][j],handle['com_offset_body_m'])
    bad = deepcopy(h['spec']);bad['fixture_nodes']=[0]
    with pytest.raises(ValueError):dataset.input_features(public,bad,'exciting_world_a_v01')


def test_incomplete_batch_rejected_before_any_label_is_opened(manifest,tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(dataset,'ROOT',tmp_path)
    p = tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    (tmp_path/'index.json').write_text(json.dumps({'manifest_sha256':dataset.sha(p),'completed':False}))
    with pytest.raises(ValueError,match='complete batch'):
        dataset.prepare('manifest.json','index.json','output')
    assert not (tmp_path/'output').exists()
