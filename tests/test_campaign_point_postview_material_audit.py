import numpy as np
import pytest
from experiments.campaign_point_postview_material_audit import summarize_postview_material,STATE_NAMES


def fixture():
    theta=np.tile(np.array([[1000.,400.,.30,.4],[2000.,800.,.40,.8]]),(1024,1))
    ids=np.tile(np.array([0,1]),1024)
    weights={k:np.full(2048,1/2048) for k in STATE_NAMES}
    return theta,ids,weights


def test_same_empirical_set_narrower_but_wrong_is_reported_not_repaired():
    x,ids,w=fixture();w['fixed_view_0']=np.where(ids==1,1/1024,0.)
    r=summarize_postview_material(x,ids,w,[1000.,400.,.30,.4],0,np.arange(55))
    base,after=r['states']['no_additional_view'],r['states']['fixed_view_0']
    for a,b in zip(base['material'],after['material']):
        assert a['truth_in_90_percent_interval'] and a['interval_width']>0
        assert b['interval_width']==0 and not b['truth_in_90_percent_interval']
    assert after['support_true_node_probability']==0 and after['support_mode_probability']==1
    assert after['empirical_ESS']==1024 and base['empirical_ESS']==2048
    assert len(after['support_categories'])==55 and after['unique_geometry_ids_in_common_particles']==2
    assert after['positive_weight_geometry_ids']==1


def test_weighted_moments_quantiles_and_zero_mass_support_rank():
    x,ids,w=fixture();w['fixed_view_1']=np.where(ids==0,.75/1024,.25/1024)
    r=summarize_postview_material(x,ids,w,[1500.,600.,.35,.6],54,np.arange(55))
    row=r['states']['fixed_view_1'];m=row['material'][0]
    assert m['mean']==1250 and m['std']==pytest.approx(np.sqrt(.75*.25)*1000)
    assert (m['quantile05'],m['median'],m['quantile95'])==(1000,1000,2000)
    assert row['support_true_node_probability']==0 and row['support_true_node_rank']==55
    assert row['support_categories'][-1]['node_id']==54


def test_identical_weights_leave_every_state_identical_and_inputs_unchanged():
    x,ids,w=fixture();original=x.copy();weights={k:v.copy() for k,v in w.items()}
    r=summarize_postview_material(x,ids,w,[1500.,600.,.35,.6],0,np.arange(55))
    assert all(r['states'][k]==r['states']['no_additional_view'] for k in STATE_NAMES)
    np.testing.assert_array_equal(x,original)
    for k in STATE_NAMES:np.testing.assert_array_equal(w[k],weights[k])


def test_missing_categories_or_invalid_normalization_are_rejected_not_repaired():
    x,ids,w=fixture()
    with pytest.raises(ValueError):summarize_postview_material(x,ids,w,[1500.,600.,.35,.6],0,np.arange(54))
    w['fixed_view_2'][0]=-.1
    with pytest.raises(ValueError):summarize_postview_material(x,ids,w,[1500.,600.,.35,.6],0,np.arange(55))
    x,ids,w=fixture();w['no_additional_view']=np.where(ids==1,1/1024,0.)
    with pytest.raises(ValueError,match='SAMEequal'):summarize_postview_material(x,ids,w,[1500.,600.,.35,.6],0,np.arange(55))


def test_paired_policy_summary_counts_narrower_but_lost_coverage():
    from experiments.campaign_point_postview_material_audit import summarize_policy_comparisons
    x,ids,w=fixture();w['fixed_view_0']=np.where(ids==1,1/1024,0.)
    diagnostic=summarize_postview_material(x,ids,w,[1000.,400.,.30,.4],0,np.arange(55))
    r=summarize_policy_comparisons([dict(diagnostic=diagnostic,selected_active_view=0,selected_random_view=1)],best_fixed=0)
    for m in r['active']['material']:
        assert m['narrower_and_loses_previous_coverage_objects']==1 and m['coverage_lost_objects']==1
        assert m['paired_mean_width_change_from_noaction']<0 and m['covered']==0
    assert r['random']==r['no_additional_view']
    assert r['best_fixed_on_calibration']==r['active']


def _saved_metadata_fixture(tmp_path,monkeypatch):
    import json
    from experiments import campaign_point_postview_material_audit as audit
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    execution=dict(analysis_output='analysis',source_sha256={'frozen.py':'sourcehash'})
    objects=[]
    for index in range(16):
        obj=dict(index=index,object_id=f'point_prior_{index:03d}',role='CALIBRATION',realized_hypothesis=dict(path=f'PRIVATE/{index}.json',sha256='MUST_NOT_OPEN'))
        objects.append(obj);directory=tmp_path/'analysis/objects'/obj['object_id'];directory.mkdir(parents=True)
        for name in ['predictive-states.npz','empirical-posterior.npz','early-observation.npz']:(directory/name).write_bytes(b'NOT_AN_ARRAY')
        identity=dict(object_index=index,role='CALIBRATION',method_sha256=audit.METHOD_SHA,execution_protocol_sha256='executionhash')
        choice=dict(**identity,source_sha256=execution['source_sha256'],actual_frame24_opened=False,selected_index=0,selected_random_view=1,
            empirical_posterior=audit._record(directory/'empirical-posterior.npz'),early_observation=audit._record(directory/'early-observation.npz'))
        audit._write(directory/'active-choice-frozen.json',choice)
        marker=dict(**identity,source_sha256=execution['source_sha256'],state_names=list(STATE_NAMES),future_A_or_B_truth_opened=False,material_or_support_truth_opened=False,
            states=audit._record(directory/'predictive-states.npz'),active_choice=audit._record(directory/'active-choice-frozen.json'))
        audit._write(directory/'posterior-states-frozen.json',marker)
        result=dict(**identity,object_id=obj['object_id'],completed=True,posterior_frozen_before_future_truth=True,
            posterior_freeze=audit._record(directory/'posterior-states-frozen.json'),active_choice=marker['active_choice'],states={n:{} for n in STATE_NAMES},selected_active_view=0,selected_random_view=1)
        audit._write(directory/'result.json',result)
    status=dict(role='CALIBRATION',terminal=True,completed=True,method_sha256=audit.METHOD_SHA,execution_protocol_sha256='executionhash',
        planned_object_indices=list(range(16)),completed_object_indices=list(range(16)),unlaunched_object_indices=[],failures=[])
    audit._write(tmp_path/'analysis/calibration-execution.json',status)
    return audit,objects,execution


def test_all16_markers_preflight_before_arrays_or_private_hypotheses(tmp_path,monkeypatch):
    audit,objects,execution=_saved_metadata_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Metadata preflight must not openarrays'))
    records,status=audit.preflight_saved_objects(objects,execution,'executionhash','CALIBRATION')
    assert len(records)==16 and all(r['object']['role']=='CALIBRATION' for r in records)
    assert status['path']=='analysis/calibration-execution.json'
    with pytest.raises(ValueError,match='Exact16/32'):audit.preflight_saved_objects(objects,execution,'executionhash','RESERVED_EVALUATION')


def test_failed_last_marker_stops_before_any_case_result_value_reads(tmp_path,monkeypatch):
    audit,objects,execution=_saved_metadata_fixture(tmp_path,monkeypatch)
    path=tmp_path/'analysis/objects/point_prior_015/posterior-states-frozen.json'
    marker=audit._read(path);marker['future_A_or_B_truth_opened']=True;audit._write(path,marker)
    original=audit._read
    def guarded_read(path):
        if str(path).endswith('/result.json'):pytest.fail('No resultvalues before ALLmarkers verified')
        return original(path)
    monkeypatch.setattr(audit,'_read',guarded_read)
    with pytest.raises(ValueError,match='frozenposterior'):audit.preflight_saved_objects(objects,execution,'executionhash','CALIBRATION')


def test_incomplete_global_stage_rejects_before_case_metadata(tmp_path,monkeypatch):
    audit,objects,execution=_saved_metadata_fixture(tmp_path,monkeypatch)
    path=tmp_path/'analysis/calibration-execution.json';status=audit._read(path);status['completed']=False;audit._write(path,status)
    original=audit._read
    def guarded_read(path):
        if '/objects/' in str(path):pytest.fail('No casefiles before stagecomplete')
        return original(path)
    monkeypatch.setattr(audit,'_read',guarded_read)
    with pytest.raises(ValueError,match='ALL selectedrole'):audit.preflight_saved_objects(objects,execution,'executionhash','CALIBRATION')


def test_offline_cli_synthetic_end_to_end_after_preflight_and_no_reserved_bypass(tmp_path,monkeypatch):
    from pathlib import Path
    from experiments import run_point_prior_analysis as old_runner
    from experiments import summarize_point_prior_analysis as old_summary
    audit,objects,execution=_saved_metadata_fixture(tmp_path,monkeypatch)
    own=tmp_path/'experiments/campaign_point_postview_material_audit.py';own.parent.mkdir();own.write_text('synthetic audit source')
    monkeypatch.setattr(audit,'__file__',str(own))
    prior=tmp_path/'prior.json';audit._write(prior,dict(masks=[dict(node_id=i) for i in range(55)]));execution['support_prior']=audit._record(prior)
    for obj in objects:
        path=tmp_path/f"PRIVATE/{obj['index']}.json";path.parent.mkdir(exist_ok=True)
        audit._write(path,dict(boundary_kind='point_pin',fixture_nodes=[0],material=dict(zip(audit.MATERIAL_KEYS,[1000.,400.,.30,.4]))))
        obj['realized_hypothesis']=audit._record(path)
    recipe=dict(schema='point-postview-material-audit-recipe-v0.1',method_sha256=audit.METHOD_SHA,
        roles=['CALIBRATION','RESERVED_EVALUATION'],particles=2048,states=list(STATE_NAMES),execute_authorization='ROOT_EXPLICIT_SEPARATE_REQUIRED',
        source_sha256={own.relative_to(tmp_path).as_posix():audit._sha(own)},execution_protocol_sha256='executionhash',output_prefix='audit-output')
    rp=tmp_path/'recipe.json';audit._write(rp,recipe)
    monkeypatch.setattr(old_runner,'validate_execution',lambda *a,**k:(execution,{},{}))
    def eligible(plan,role,**kwargs):
        assert role=='CALIBRATION' and kwargs['calibration_freeze'] is None
        return objects
    monkeypatch.setattr(old_runner,'eligible_objects',eligible)
    monkeypatch.setattr(old_summary,'validate_freeze',lambda *a,**k:dict(best_fixed_view_index=2))
    done=False;original=audit.preflight_saved_objects
    def preflight(*a,**k):
        nonlocal done
        value=original(*a,**k);done=True;return value
    monkeypatch.setattr(audit,'preflight_saved_objects',preflight)
    x,ids,w=fixture()
    class Loaded:
        def __init__(self,values):self.values=values
        def __enter__(self):return self.values
        def __exit__(self,*a):pass
    def load(path,**kwargs):
        assert done,'Noarrays before entireselectedrole preflight'
        if str(path).endswith('empirical-posterior.npz'):return Loaded(dict(physical_coordinates=x,geometry_ids=ids))
        assert str(path).endswith('predictive-states.npz')
        return Loaded({k+'__weights':v for k,v in w.items()})
    monkeypatch.setattr(np,'load',load)
    result=audit.run_offline_audit('recipe.json',audit._sha(rp),'execution.json','CALIBRATION',calibration_record={'path':'frozen.json','sha256':'fakevalid'})
    value=audit._read(tmp_path/result['path'])
    assert value['completed'] and value['objects']==16 and 'best_fixed_on_calibration' in value['policies']
    assert value['same2048_particle_comparison'] and value['C28_full_reference_intervals_unchanged']
    with pytest.raises(ValueError,match='explicit hash-bound release'):
        audit.run_offline_audit('recipe.json',audit._sha(rp),'execution.json','RESERVED_EVALUATION')
