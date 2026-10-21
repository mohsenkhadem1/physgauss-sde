"""Synthetic v0.2 role barriers, freeze bindings and object-level statistics."""
import json
import numpy as np
import pytest
from experiments import summarize_point_rotation_analysis as report


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    own=tmp_path/'experiments/summarize_point_rotation_analysis.py';own.parent.mkdir();own.write_text('frozen synthetic aggregator')
    monkeypatch.setattr(report,'__file__',str(own))
    (own.parent/'campaign_simultaneous_bands.py').write_text('frozen synthetic band source')
    method=tmp_path/'method.json';dump(method,{'schema':'synthetic new method'})
    plan={'objects':[dict(index=i,object_id=f'fresh_rotation_{i:03d}',role='CALIBRATION' if i<16 else 'RESERVED_EVALUATION') for i in range(48)]}
    output=tmp_path/'analysis';output.mkdir()
    execution=tmp_path/'execution.json'
    dump(execution,dict(schema='point-rotation-analysis-execution-v0.2',method=report.record(method),analysis_output='analysis',
        source_sha256={'experiments/'+p.name:report.sha(p) for p in own.parent.iterdir()}))
    digest=report.sha(execution)
    def validate(path,expected):
        path=tmp_path/path
        if report.sha(path)!=expected:raise ValueError('Execution changed')
        return report.read(path),report.read(method),plan
    monkeypatch.setattr(report,'validate_execution_inputs',validate)
    def eligible(plan,role,execution_sha,method_sha,**kwargs):
        assert execution_sha==digest and method_sha==report.sha(method)
        if role=='RESERVED_EVALUATION':
            assert kwargs['release']['calibration_freeze']==kwargs['calibration_freeze_record']
            assert kwargs['calibration_freeze']['completed'] is True
        return [o for o in plan['objects'] if o['role']==role]
    monkeypatch.setattr(report,'eligible',eligible)
    def episode(i):
        obj=plan['objects'][i];directory=output/'objects'/obj['object_id'];directory.mkdir(parents=True,exist_ok=True)
        ex=report.read(execution)
        common=dict(object_index=i,role=obj['role'],method_sha256=ex['method']['sha256'],execution_protocol_sha256=digest,source_sha256=ex['source_sha256'])
        choice=dict(**common,actual_frame24_opened=False,selected_index=2,selected_random_view=0)
        dump(directory/'active-choice-frozen.json',choice)
        # Deliberately not an NPZ: aggregation must hash but never parse responses.
        (directory/'predictive-states.npz').write_bytes(b'opaque synthetic predictive states')
        marker=dict(**common,active_choice=report.record(directory/'active-choice-frozen.json'),states=report.record(directory/'predictive-states.npz'),
            state_names=list(report.STATE_NAMES),future_A_or_B_truth_opened=False,material_or_support_truth_opened=False)
        dump(directory/'posterior-states-frozen.json',marker)
        states={}
        for name in report.STATE_NAMES:
            mse=1e-6 if name=='no_additional_view' else 5e-7+(int(name[-1])-2)**2*1e-8
            states[name]=dict(A_after1s_MSE_m2=0.,A_after1p5s_MSE_m2=mse,B_all_MSE_m2=mse,
                individual_band_score=float(i%16+1),mean_target_scale_m=1e-4,view_update_ESS=200.)
        value=dict(object_index=i,object_id=obj['object_id'],role=obj['role'],completed=True,method_sha256=ex['method']['sha256'],
            execution_protocol_sha256=digest,posterior_frozen_before_future_truth=True,posterior_freeze=report.record(directory/'posterior-states-frozen.json'),
            states=states,joint_band_score=float(i%16+1),selected_active_view=2,selected_random_view=0)
        path=directory/'result.json';dump(path,value);return path
    return output,execution,digest,episode


def test_all16_freeze_dynamic_method_rank16_and_lowest_fixed_tie(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,ex,digest);freeze=report.read(tmp_path/marker['path'])
    assert freeze['schema']==report.CALIBRATION_SCHEMA
    assert freeze['band_calibration']['scale']==16. and freeze['band_calibration']['order_statistic_rank']==16
    assert freeze['method']==report.read(ex)['method'] and freeze['best_fixed_view_index']==2
    release=report.release_reserved(marker,output/'release.json')
    assert report.read(tmp_path/release['path'])['schema']==report.RELEASE_SCHEMA
    assert not (output/'objects/fresh_rotation_016').exists()


def test_missing_role_or_old_execution_rejected(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(15):episode(i)
    with pytest.raises(FileNotFoundError):report.freeze_calibration(output,ex,digest)
    path=episode(15);value=report.read(path);value['role']='RESERVED_EVALUATION';dump(path,value)
    with pytest.raises(ValueError,match='prescribed'):report.freeze_calibration(output,ex,digest)
    value=report.read(ex);value['schema']='point-prior-analysis-execution-v0.1';dump(ex,value)
    with pytest.raises(ValueError,match='rotation-aware'):report.freeze_calibration(output,ex,report.sha(ex))


def test_all_freezes_checked_before_any_result_values(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    # Invalid final state marker must stop before the deliberately malformed first result.
    (output/'objects/fresh_rotation_000/result.json').write_text('not JSON')
    path=output/'objects/fresh_rotation_015/posterior-states-frozen.json';value=report.read(path)
    value['future_A_or_B_truth_opened']=True;dump(path,value)
    with pytest.raises(ValueError,match='before future truth'):report.freeze_calibration(output,ex,digest)


@pytest.mark.parametrize('kind',['states','result','fixed','band'])
def test_modified_freeze_dependencies_block_reserved_release(tmp_path,monkeypatch,kind):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,ex,digest)
    if kind=='states':(output/'objects/fresh_rotation_000/predictive-states.npz').write_bytes(b'changed')
    elif kind=='result':
        path=output/'objects/fresh_rotation_000/result.json';v=report.read(path);v['joint_band_score']=999.;dump(path,v)
    else:
        path=tmp_path/marker['path'];v=report.read(path)
        if kind=='fixed':v['best_fixed_view_index']=1
        else:v['band_calibration']['scale']=100.
        dump(path,v);marker=report.record(path)
    with pytest.raises(ValueError):report.release_reserved(marker,output/'release.json')


def test_reserved_all32_exact_ci_fair_windows_bootstrap_and_bandwidth(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,ex,digest);release=report.release_reserved(marker,output/'release.json')
    for i in range(16,48):episode(i)
    result=report.evaluate_reserved(output,release);v=report.read(tmp_path/result['path'])
    coverage=v['joint_simultaneous_band_coverage']
    assert coverage['covered']==32 and coverage['total']==32 and coverage['exact_binomial_95_interval'][1]==1.
    assert 0<coverage['exact_binomial_95_interval'][0]<1
    assert v['prediction']['no_additional_view']['A_after1p5s_MSE_m2']['pooled_coordinate_RMSE_mm']==1.
    assert v['prediction']['active']['mean_calibrated_full_bandwidth_m']==pytest.approx(.0032)
    paired=v['paired_active_minus_baseline']['random']['A_after1p5s_MSE_m2']['MSE_difference_m2']
    assert paired['mean']<0 and paired['seed']==2026120600 and paired['replicates']==10000
    assert paired['independent_unit']=='complete object'


def test_no_partial_reserved_or_wrong_output_and_choice_mutation(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):episode(i)
    marker=report.freeze_calibration(output,ex,digest);release=report.release_reserved(marker,output/'release.json')
    for i in range(16,47):episode(i)
    with pytest.raises(FileNotFoundError):report.evaluate_reserved(output,release)
    with pytest.raises(ValueError,match='output directory'):report.evaluate_reserved(tmp_path/'wrong',release)
    path=episode(47);v=report.read(path);v['selected_active_view']=1;dump(path,v)
    with pytest.raises(ValueError,match='unchanged declared'):report.evaluate_reserved(output,release)


def test_own_source_must_be_pinned_and_paths_cannot_escape(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    v=report.read(ex);del v['source_sha256']['experiments/summarize_point_rotation_analysis.py'];dump(ex,v)
    with pytest.raises(ValueError,match='Aggregator must be pinned'):report.freeze_calibration(output,ex,report.sha(ex))
    with pytest.raises(ValueError,match='repository-relative'):report.checked(dict(path='../outside',sha256='x'))


def test_constant_paired_difference_has_exact_interval():
    r=report.paired_bootstrap(np.full(32,-2e-7))
    np.testing.assert_allclose(r['percentile_95_interval'],[-2e-7,-2e-7],rtol=0,atol=1e-20)


def test_fixed_camera_tie_is_lowest_index_without_reserved_selection(tmp_path,monkeypatch):
    output,ex,digest,episode=fixture(tmp_path,monkeypatch)
    for i in range(16):
        path=episode(i);v=report.read(path)
        for j in range(4):v['states'][f'fixed_view_{j}']['A_after1p5s_MSE_m2']=5e-7
        dump(path,v)
    marker=report.freeze_calibration(output,ex,digest)
    assert report.read(tmp_path/marker['path'])['best_fixed_view_index']==0


def test_actual_runner_wrapper_passes_canonical_relative_path(tmp_path,monkeypatch):
    from experiments import run_point_rotation_analysis as runner
    monkeypatch.setattr(report,'ROOT',tmp_path)
    captured={}
    def validate(path,digest,*,check_environment):
        captured.update(path=path,digest=digest,check_environment=check_environment)
        return {},{},{}
    monkeypatch.setattr(runner,'validate_execution',validate)
    report.validate_execution_inputs(tmp_path/'execution.json','digest')
    assert captured==dict(path='execution.json',digest='digest',check_environment=False)


def test_actual_new_runner_rejects_old_cohort_and_requires_new_reserved_release():
    from experiments import run_point_rotation_analysis as runner
    plan=dict(schema='independent-point-rotation-prior-cohort-v0.2',roles=dict(CALIBRATION=list(range(16)),RESERVED_EVALUATION=list(range(16,48))),
        objects=[dict(index=i,object_id=f'point_prior_{i:03d}',seed=2026120100+i,role='CALIBRATION' if i<16 else 'RESERVED_EVALUATION') for i in range(48)])
    selected=runner.eligible_objects(plan,'CALIBRATION',execution_sha256='e',method_sha256='m')
    assert [v['index'] for v in selected]==list(range(16))
    with pytest.raises(ValueError,match='reserved release'):
        runner.eligible_objects(plan,'RESERVED_EVALUATION',execution_sha256='e',method_sha256='m')
    plan['schema']='independent-point-prior-cohort-v0.1'
    with pytest.raises(ValueError,match='NEW independent48'):
        runner.eligible_objects(plan,'CALIBRATION',execution_sha256='e',method_sha256='m')
