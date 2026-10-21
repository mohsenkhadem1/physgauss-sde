"""Synthetic object-role, split freeze, paired-unit and exact coverage checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from experiments import summarize_point_prior_analysis as report


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    method=tmp_path/report.METHOD;dump(method,{'cohort_plan':{'synthetic':True},'base_inference':{'library_references':[]}})
    monkeypatch.setattr(report,'METHOD_SHA',report.sha(method))
    own=tmp_path/'experiments/summarize_point_prior_analysis.py';own.parent.mkdir();own.write_text('synthetic frozen source')
    monkeypatch.setattr(report,'__file__',str(own))
    (own.parent/'campaign_simultaneous_bands.py').write_text('synthetic frozen band source')
    output=tmp_path/'analysis';output.mkdir()
    names=['run_point_prior_analysis.py','summarize_point_prior_analysis.py','campaign_simultaneous_bands.py',
        'campaign_point_prior_posterior.py','campaign_point_active_views.py']
    for name in names:
        path=own.parent/name
        if not path.exists():path.write_text('synthetic source')
    execution=tmp_path/'execution.json'
    dump(execution,dict(schema='point-prior-analysis-execution-v0.1',status='FROZEN_POINT_PRIOR_ANALYSIS_EXECUTION',
        method=dict(path=report.METHOD,sha256=report.METHOD_SHA),cohort_plan={'synthetic':True},prior_libraries=[],
        roles=dict(CALIBRATION=list(range(16)),RESERVED_EVALUATION=list(range(16,48))),analysis_output='analysis',
        source_sha256={'experiments/'+name:report.sha(own.parent/name) for name in names}))
    execution_sha=report.sha(execution)
    def synthetic_execution_validator(path,digest):
        path=tmp_path/path
        assert report.sha(path)==digest
        data=report.read(path)
        for name,value in data['source_sha256'].items():
            assert report.sha(tmp_path/name)==value
        return data,{},{}
    monkeypatch.setattr(report,'validate_execution_inputs',synthetic_execution_validator)
    def episode(i):
        states={}
        for label in ['no_additional_view',*[f'fixed_view_{j}' for j in range(4)]]:
            mse=1e-6 if label=='no_additional_view' else 5e-7+(int(label[-1])-2)**2*1e-8
            states[label]=dict(A_after1s_MSE_m2=0.,A_after1p5s_MSE_m2=mse,
                B_all_MSE_m2=mse,individual_band_score=float(i%16+1),mean_target_scale_m=1e-4)
        value=dict(object_index=i,role='CALIBRATION' if i<16 else 'RESERVED_EVALUATION',completed=True,
            method_sha256=report.METHOD_SHA,execution_protocol_sha256=execution_sha,
            posterior_frozen_before_future_truth=True,states=states,joint_band_score=float(i%16+1),
            selected_active_view=2,selected_random_view=0)
        path=output/'objects'/f'point_prior_{i:03d}'/'result.json';dump(path,value)
        return path
    return output,execution,execution_sha,episode


def test_sixteen_complete_objects_freeze_scale_and_fixed_camera_before_release(tmp_path,monkeypatch):
    output,execution,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,execution,digest)
    data=report.read(tmp_path/marker['path'])
    assert data['band_calibration']['scale']==16. and data['best_fixed_view_index']==2
    assert data['calibration_object_indices']==list(range(16))
    release=report.release_reserved(marker,output/'release.json')
    assert report.read(tmp_path/release['path'])['object_indices']==list(range(16,48))
    # No reserved result files need exist for release; there was no test access.
    assert not (output/'objects/point_prior_016').exists()


def test_missing_or_role_swapped_objects_are_not_filtered(tmp_path,monkeypatch):
    output,execution,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(15):episode(i)
    with pytest.raises(FileNotFoundError):report.freeze_calibration(output,execution,digest)
    path=episode(15);value=report.read(path);value['role']='RESERVED_EVALUATION';dump(path,value)
    with pytest.raises(ValueError,match='prescribed'):report.freeze_calibration(output,execution,digest)


def test_modified_calibration_result_blocks_reserved_release(tmp_path,monkeypatch):
    output,execution,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,execution,digest)
    path=output/'objects/point_prior_000/result.json';value=report.read(path);value['joint_band_score']=999.;dump(path,value)
    with pytest.raises(ValueError,match='changed'):report.release_reserved(marker,output/'release.json')


def test_object_paired_bootstrap_and_full_coverage_interval_are_not_frame_replicates(tmp_path,monkeypatch):
    output,execution,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,execution,digest)
    release=report.release_reserved(marker,output/'release.json')
    for i in range(16,48):episode(i)
    result=report.evaluate_reserved(output,release);value=report.read(tmp_path/result['path'])
    coverage=value['joint_simultaneous_band_coverage']
    assert coverage['covered']==32 and coverage['total']==32
    assert 0<coverage['exact_binomial_95_interval'][0]<1.
    assert coverage['exact_binomial_95_interval'][1]==1.
    comparison=value['paired_active_minus_baseline']['random']['A_after1p5s_MSE_m2']['MSE_difference_m2']
    assert comparison['mean']<0 and comparison['independent_unit']=='complete object'
    # The deliberately zero A_after1s MSE was not used for action comparison.
    assert value['prediction']['no_additional_view']['A_after1p5s_MSE_m2']['pooled_coordinate_RMSE_mm']==1.


def test_constant_paired_object_difference_has_exact_constant_interval():
    result=report.paired_bootstrap(np.full(32,-2e-7))
    np.testing.assert_allclose(result['percentile_95_interval'],[-2e-7,-2e-7],rtol=0,atol=1e-20)
