"""Development-only raw export -> full predictor future-target isolation regression."""
from pathlib import Path
import importlib
import numpy as np
import h5py
import pytest
from scripts.prefix_ridge_switch_v13.inputs import softvt_payload, save_payload, load_payload, load_scene_payload

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('module',['softvt_slip','softvt_overlap','softvt_predict'])
@pytest.mark.parametrize('anchor',['contact_scene_state','model_selection_selected'])
def test_development_future_object_mutation_preserves_export_and_forecast(tmp_path,module,anchor):
    # Fixed development case only. Complete controls are permitted inputs; future
    # object states are adversarially changed before the prefix exporter runs.
    name='object-soft-task0-demo0'
    folder=ROOT/'runs/external-benchmark-v11/raw/SoftVTBench/object-soft/libero_object/libero_object_task0/replayed_demos'
    with h5py.File(next(folder.glob('*.hdf5'))) as f:
        g=f['data/demo_0'];asset=g.attrs['asset_name']
        raw=dict(object_prefix=g[f'states/deformable_object/{asset}/nodal_pos_w'][:],eef_pose=g['obs/eef_pose'][:],robot_root=g['initial_state/articulation/robot/root_pose'][0],width=g['obs/gripper_width'][:],command=g['obs/gripper_binary'][:])
    end=int(.6*len(raw['eef_pose']))
    changed={k:np.array(v,copy=True) for k,v in raw.items()}
    changed['object_prefix'][end:]=np.random.default_rng(14).normal(1e6,1e3,changed['object_prefix'][end:].shape)
    one=softvt_payload(raw,'object-soft',0,0,end=end)
    two=softvt_payload(changed,'object-soft',0,0,end=end)
    assert 'evaluation' not in one and 'evaluation' not in two
    for key in one:
        if isinstance(one[key],np.ndarray):np.testing.assert_array_equal(one[key],two[key])
        else:assert one[key]==two[key]
    save_payload(tmp_path/'a.npz',one);save_payload(tmp_path/'b.npz',two)
    one=load_payload(tmp_path/'a.npz');two=load_payload(tmp_path/'b.npz')
    scene=load_scene_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs'/name/'scene.npz')
    old=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'/name/'predictions.npz'
    if anchor=='contact_scene_state':old=ROOT/'runs/contact-benchmark-v12/evaluation/development'/name/'predictions.npz'
    with np.load(old) as z:base=z[anchor].copy()
    predictor=importlib.import_module('scripts.benchmark_improvement_v14.'+module).predict
    a,ma=predictor(one,base,dict(scene=scene));b,mb=predictor(two,base,dict(scene=scene))
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)
    assert ma==mb
