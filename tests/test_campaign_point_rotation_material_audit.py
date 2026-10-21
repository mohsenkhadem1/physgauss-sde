"""Synthetic seal barriers and retained negative material/support diagnostics."""
from pathlib import Path
import numpy as np
import pytest
from experiments import campaign_point_rotation_material_audit as audit


def toy():
    x=np.tile([[1000.,400.,.30,.4],[2000.,800.,.40,.8]],(1024,1)).astype(np.float64)
    ids=np.tile([0,1],1024);weights={name:np.full(2048,1/2048) for name in audit.STATE_NAMES}
    weights['fixed_view_0']=np.where(ids==1,1/1024,0.)
    return x,ids,weights


def test_negative_narrower_confidently_wrong_and_MAE_are_retained():
    x,ids,weights=toy();truth=[1000.,400.,.30,.4]
    diagnostic=audit.math_audit.summarize_postview_material(x,ids,weights,truth,0,np.arange(55))
    objects=[dict(object_index=16,material_truth=truth,diagnostic=diagnostic,selected_active_view=0,selected_random_view=1)]
    result=audit.summarize_comparisons(objects,best_fixed=0)
    base,active=result['no_additional_view'],result['active']
    assert active['material'][0]['covered']==0 and active['material'][0]['mean_interval_width']==0
    assert active['material'][0]['narrower_and_loses_previous_coverage_objects']==1
    assert active['material'][0]['posterior_mean_MAE']==1000 and base['material'][0]['posterior_mean_MAE']==500
    assert active['material'][0]['paired_mean_absolute_error_change_from_noaction']==500
    assert active['support_mode_confidence']['confidently_wrong_at_least_90_percent_objects']==1
    assert active['true_support_probability']['exactzero_objects']==1
    assert active['empirical_row_ESS']['minimum']==1024
    assert result['random']==base and result['best_fixed_on_calibration']==active
    assert len(diagnostic['states']['fixed_view_0']['support_categories'])==55


def fixture(tmp_path,monkeypatch,role='CALIBRATION'):
    monkeypatch.setattr(audit,'ROOT',tmp_path);monkeypatch.setattr(audit.aggregate,'ROOT',tmp_path)
    indices=list(range(16)) if role=='CALIBRATION' else list(range(16,48))
    execution=dict(analysis_output='analysis',method=dict(path='method.json',sha256='methodhash'),source_sha256={'synthetic.py':'sourcehash'})
    objects=[]
    for i in indices:
        obj=dict(index=i,object_id=f'new_point_{i:03d}',role=role,realized_hypothesis=dict(path=f'PRIVATE/{i}.json',sha256='not-read'))
        objects.append(obj);directory=tmp_path/'analysis/objects'/obj['object_id'];directory.mkdir(parents=True)
        for filename in ['predictive-states.npz','empirical-posterior.npz']:(directory/filename).write_bytes(b'OPAQUE_NOT_AN_ARRAY')
        identity=dict(object_index=i,role=role,method_sha256='methodhash',execution_protocol_sha256='exechash')
        choice=dict(**identity,source_sha256=execution['source_sha256'],actual_frame24_opened=False,selected_index=0,selected_random_view=1)
        audit.write(directory/'active-choice-frozen.json',choice)
        marker=dict(**identity,source_sha256=execution['source_sha256'],state_names=list(audit.STATE_NAMES),future_A_or_B_truth_opened=False,material_or_support_truth_opened=False,
            states=audit.record(directory/'predictive-states.npz'),active_choice=audit.record(directory/'active-choice-frozen.json'),empirical_posterior=audit.record(directory/'empirical-posterior.npz'))
        audit.write(directory/'posterior-states-frozen.json',marker)
        states={name:dict(A_after1s_MSE_m2=0.,A_after1p5s_MSE_m2=1e-6,B_all_MSE_m2=1e-6,individual_band_score=1.) for name in audit.STATE_NAMES}
        result=dict(**identity,object_id=obj['object_id'],completed=True,posterior_frozen_before_future_truth=True,posterior_freeze=audit.record(directory/'posterior-states-frozen.json'),
            states=states,joint_band_score=1.,selected_active_view=0,selected_random_view=1)
        audit.write(directory/'result.json',result)
    status=dict(role=role,terminal=True,completed=True,method_sha256='methodhash',execution_protocol_sha256='exechash',
        planned_object_indices=indices,completed_object_indices=indices,unlaunched_object_indices=[],failures=[])
    statuspath=tmp_path/'analysis'/('calibration-execution.json' if role=='CALIBRATION' else 'reserved-execution.json');audit.write(statuspath,status)
    return objects,execution,statuspath


@pytest.mark.parametrize('role',['CALIBRATION','RESERVED_EVALUATION'])
def test_all_role_freezes_verified_without_arrays_or_private_values(tmp_path,monkeypatch,role):
    objects,execution,status=fixture(tmp_path,monkeypatch,role)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Metadata preflight must never parse NPZ'))
    records,_=audit.preflight_saved_objects(objects,execution,'exechash',role)
    assert len(records)==(16 if role=='CALIBRATION' else 32)
    assert records[-1]['selected_active_view']==0 and records[-1]['empirical']['path'].endswith('/empirical-posterior.npz')
    assert not (tmp_path/'PRIVATE').exists()


def test_incomplete_stage_stops_before_casefiles(tmp_path,monkeypatch):
    objects,execution,path=fixture(tmp_path,monkeypatch);v=audit.read(path);v['completed']=False;audit.write(path,v)
    original=audit.read
    def read(p):
        if '/objects/' in str(p):pytest.fail('No object metadata before whole stage complete')
        return original(p)
    monkeypatch.setattr(audit,'read',read)
    with pytest.raises(ValueError,match='Entire selected-role'):audit.preflight_saved_objects(objects,execution,'exechash','CALIBRATION')


def test_last_corrupt_empirical_hash_stops_before_any_result_value(tmp_path,monkeypatch):
    objects,execution,_=fixture(tmp_path,monkeypatch)
    (tmp_path/'analysis/objects/new_point_015/empirical-posterior.npz').write_bytes(b'corrupted')
    original=audit.aggregate.read
    def read(p):
        if str(p).endswith('/result.json'):pytest.fail('No result values before ALL empirical hashes')
        return original(p)
    monkeypatch.setattr(audit.aggregate,'read',read)
    with pytest.raises(ValueError,match='changed'):audit.preflight_saved_objects(objects,execution,'exechash','CALIBRATION')


def test_missing_calibration_freeze_rejected_before_execution_or_case_reads(monkeypatch):
    monkeypatch.setattr(audit.aggregate,'check_execution',lambda *a,**k:pytest.fail('No execution reads before required calibration seal supplied'))
    with pytest.raises(ValueError,match='Completed16'):audit.gate_stage({},'CALIBRATION',None)


def mock_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    execution_record=dict(path='exec.json',sha256='exechash');method=dict(path='method.json',sha256='methodhash')
    calibration=dict(path='calibration.json',sha256='calibhash');recipe=dict(execution_protocol=execution_record,method=method)
    freeze=dict(method=method,best_fixed_view_index=2,calibration_results=[])
    monkeypatch.setattr(audit,'checked',lambda r:tmp_path/r['path'])
    monkeypatch.setattr(audit.aggregate,'check_execution',lambda *a,**k:(dict(method=method),{},{}))
    monkeypatch.setattr(audit.aggregate,'validate_freeze',lambda *a,**k:freeze)
    return recipe,calibration


def test_reserved_needs_primary_aggregate_not_just_release(tmp_path,monkeypatch):
    recipe,calibration=mock_gate(tmp_path,monkeypatch)
    monkeypatch.setattr(audit,'preflight_saved_objects',lambda *a,**k:pytest.fail('No perobject preflight until full primary seal'))
    with pytest.raises(ValueError,match='Completed32'):audit.gate_stage(recipe,'RESERVED_EVALUATION',calibration,release_record=dict(path='release.json',sha256='r'))


def test_old_release_and_incomplete_primary_are_rejected(tmp_path,monkeypatch):
    recipe,calibration=mock_gate(tmp_path,monkeypatch);release_rec=dict(path='release.json',sha256='r');primary_rec=dict(path='primary.json',sha256='p')
    release=dict(schema='point-prior-reserved-release-v0.1',status='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE',method=recipe['method'],method_sha256='methodhash',execution_protocol_sha256='exechash',
        object_indices=list(range(16,48)),calibration_freeze=calibration,no_reserved_response_or_truth_used_for_calibration=True)
    audit.write(tmp_path/'release.json',release)
    with pytest.raises(ValueError,match='Exact new reserved release'):audit.gate_stage(recipe,'RESERVED_EVALUATION',calibration,release_record=release_rec,primary_record=primary_rec)
    release['schema']=audit.aggregate.RELEASE_SCHEMA;audit.write(tmp_path/'release.json',release)
    audit.write(tmp_path/'primary.json',dict(schema='point-rotation-reserved-view-evaluation-v0.2',completed=False,objects=32))
    with pytest.raises(ValueError,match='Completed hash-bound32'):audit.gate_stage(recipe,'RESERVED_EVALUATION',calibration,release_record=release_rec,primary_record=primary_rec)


def test_recipe_freezes_source_choices_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    for name in audit.SOURCE_NAMES:
        p=tmp_path/'experiments'/name;p.parent.mkdir(exist_ok=True);p.write_text('synthetic frozen source')
    rec=audit.prepare_recipe('recipe.json',dict(path='exec.json',sha256='e'),dict(path='method.json',sha256='m'),'audit')
    value=audit.validate_recipe(rec['path'],rec['sha256']);assert value['choices']['confidence_threshold']==.9
    with pytest.raises(ValueError,match='retained'):audit.prepare_recipe('recipe.json',{}, {},'audit')
    (tmp_path/'experiments'/audit.SOURCE_NAMES[1]).write_text('changed')
    with pytest.raises(ValueError,match='source closure'):audit.validate_recipe(rec['path'],rec['sha256'])


def test_synthetic_full_audit_opens_arrays_only_after_complete_preflight(tmp_path,monkeypatch):
    objects,execution,_=fixture(tmp_path,monkeypatch)
    for obj in objects:
        p=tmp_path/f"PRIVATE/{obj['index']}.json";p.parent.mkdir(exist_ok=True)
        audit.write(p,dict(boundary_kind='point_pin',fixture_nodes=[0],material=dict(zip(audit.MATERIAL_KEYS,[1000.,400.,.30,.4]))))
        obj['realized_hypothesis']=audit.record(p)
    recipe=dict(output_prefix='audit',execution_protocol=dict(path='exec.json',sha256='exechash'),method=execution['method'],source_sha256={})
    monkeypatch.setattr(audit,'validate_recipe',lambda *a:recipe)
    done=False
    def gate(*args,**kwargs):
        nonlocal done
        records,status=audit.preflight_saved_objects(objects,execution,'exechash','CALIBRATION');done=True
        return records,status,dict(best_fixed_view_index=0),np.arange(55)
    monkeypatch.setattr(audit,'gate_stage',gate)
    x,ids,weights=toy()
    class Loaded:
        def __enter__(self):return dict(physical_coordinates=x,geometry_ids=ids,**{k+'__weights':v for k,v in weights.items()})
        def __exit__(self,*a):pass
    def load(path,**kwargs):
        assert done,'ALL role markers before ANY arrays';return Loaded()
    monkeypatch.setattr(np,'load',load)
    # Only the synthetic run's declared recipe record is needed in its summary.
    (tmp_path/'recipe.json').write_text('synthetic recipe')
    result=audit.run_offline_audit('recipe.json','r','CALIBRATION',calibration_record=dict(path='calibration.json',sha256='c'))
    summary=audit.read(tmp_path/result['path'])
    assert summary['completed'] and summary['objects']==16 and summary['all_object_indices']==list(range(16))
    assert summary['policies']['active']['material'][0]['coverage_lost_objects']==16
    assert summary['policies']['active']['support_mode_confidence']['confidently_wrong_at_least_90_percent_objects']==16
    assert len(summary['prior_geometry_ids_including_zeros'])==55
    with pytest.raises(ValueError,match='retained'):audit.run_offline_audit('recipe.json','r','CALIBRATION',calibration_record={})
