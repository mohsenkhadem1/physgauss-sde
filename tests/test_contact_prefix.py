"""Calibration isolation tests; external fixtures are local and never fetched."""
import copy
from pathlib import Path
import numpy as np
import pytest
from scripts.contact_benchmark_v12.data import trim_prefix
from scripts.external_benchmark.data import softvt


def local_data(subset):
    try:return softvt(subset,0,1)
    except (FileNotFoundError,StopIteration):pytest.skip('Local SoftVT development fixture unavailable')


@pytest.mark.parametrize('subset',['object-soft','spatial-soft'])
def test_full_prefix_reproduces_frozen_adapter(subset):
    data=local_data(subset);out=trim_prefix(data,data['train_end'])
    for key in ('observed','observed_mask','controller_targets','attachment_weights','attachment_active'):
        np.testing.assert_allclose(out[key],data[key],atol=1e-12,rtol=1e-12)
    assert out['grasp']==data['grasp']


@pytest.mark.parametrize('subset',['object-soft','spatial-soft'])
def test_short_calibration_ignores_later_observed_object_motion_and_old_calibration(subset):
    data=local_data(subset);cut=int(data['train_end']*.7)
    expected=trim_prefix(data,cut);poison=copy.deepcopy(data)
    poison['observed'][cut:]=1e8
    poison['observed_mask'][cut:]=False
    poison['controller_targets'][:]=np.nan
    poison['attachment_weights'][:]=np.nan
    poison['attachment_active'][:]=np.nan
    poison['grasp']=len(data['times'])-1
    poison['evaluation']['points'][:]=np.nan
    actual=trim_prefix(poison,cut)
    for key in ('observed','observed_mask','controller_targets','attachment_weights','attachment_active'):
        np.testing.assert_array_equal(actual[key],expected[key])
    assert actual['grasp']<cut


def test_short_calibration_reads_only_tool_pose_from_raw_source(tmp_path):
    import h5py
    data=local_data('object-soft');cut=int(data['train_end']*.7);expected=trim_prefix(data,cut)
    source=tmp_path/'tools_only.hdf5'
    with h5py.File(data['source']) as src,h5py.File(source,'w') as dst:
        for key in ('obs/eef_pose','initial_state/articulation/robot/root_pose'):
            dst.create_dataset('data/demo_1/'+key,data=src['data/demo_1/'+key][:])
    minimal=copy.deepcopy(data);minimal['source']=str(source)
    actual=trim_prefix(minimal,cut)
    np.testing.assert_array_equal(actual['controller_targets'],expected['controller_targets'])
    assert len(actual['observed'])==cut
