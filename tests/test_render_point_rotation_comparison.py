"""Synthetic-only seal, fixed selection, timestamps and true-scale video checks."""
import numpy as np
import pytest
from experiments import render_point_rotation_comparison as renderer


def values():
    result={}
    for i in range(16,48):
        result[i]=dict(object_index=i,selected_active_view=0,states={
            'no_additional_view':dict(A_after1p5s_MSE_m2=4e-6),
            'fixed_view_0':dict(A_after1p5s_MSE_m2=3e-6)})
    return result


def test_selection_uses_signed_sealed_MSE_stable_lowest_index_and_no_arrays(monkeypatch):
    v=values();v[20]['states']['fixed_view_0']['A_after1p5s_MSE_m2']=1e-6
    v[21]['states']['fixed_view_0']['A_after1p5s_MSE_m2']=8e-6
    v[22]['states']['fixed_view_0']['A_after1p5s_MSE_m2']=8e-6
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Selection cannot read prediction or truth arrays'))
    selected=renderer.select_cases(v)
    assert [r['object_index'] for r in selected]==[20,21,21]
    assert [r['selection'] for r in selected]==list(renderer.SELECTIONS)
    assert selected[1]['regression_MSE_m2']==4e-6
    # Every role survives a duplicate case; downstream renders one pair per unique index.
    assert len(selected)==3 and len({r['object_index'] for r in selected})==2
    assert [r['object_index'] for r in renderer.select_cases(values())]==[16,16,16]


def test_no_filtering_missing_case_or_nonfinite_error():
    v=values();del v[47]
    with pytest.raises(ValueError,match='Exact32'):renderer.select_cases(v)
    v=values();v[47]['states']['fixed_view_0']['A_after1p5s_MSE_m2']=float('nan')
    with pytest.raises(ValueError,match='Finite sealed'):renderer.select_cases(v)


def test_missing_incomplete_or_old_primary_seal_cannot_select_or_read_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(renderer,'ROOT',tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No arrays before completed new primary seal'))
    recipe=dict(execution_protocol=dict(sha256='e'),method=dict(sha256='m'))
    with pytest.raises(FileNotFoundError):renderer.gate_primary(recipe,dict(path='missing.json',sha256='bad'))
    path=tmp_path/'bad.json'
    for schema,complete in [('point-rotation-reserved-view-evaluation-v0.2',False),('point-prior-reserved-view-evaluation-v0.1',True)]:
        renderer.write(path,dict(schema=schema,completed=complete,objects=32,method_sha256='m',execution_protocol_sha256='e'))
        with pytest.raises(ValueError,match='Final completed NEW32'):renderer.gate_primary(recipe,renderer.record(path))


def test_gate_requires_terminal32_and_exact_all_result_records(tmp_path,monkeypatch):
    monkeypatch.setattr(renderer,'ROOT',tmp_path)
    method=dict(path='method.json',sha256='m');execution_rec=dict(path='execution.json',sha256='e');calibration=dict(path='calibration.json',sha256='c')
    release=dict(schema=renderer.aggregate.RELEASE_SCHEMA,status='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE',method=method,method_sha256='m',execution_protocol_sha256='e',
        object_indices=list(range(16,48)),calibration_freeze=calibration,no_reserved_response_or_truth_used_for_calibration=True)
    renderer.write(tmp_path/'release.json',release);release_rec=renderer.record(tmp_path/'release.json')
    result_records=[dict(path=f'analysis/objects/{i}/result.json',sha256=str(i)) for i in range(16,48)]
    evaluation=dict(schema='point-rotation-reserved-view-evaluation-v0.2',completed=True,objects=32,method_sha256='m',execution_protocol_sha256='e',
        release=release_rec,calibration_freeze=calibration,reserved_results=result_records)
    renderer.write(tmp_path/'evaluation.json',evaluation)
    recipe=dict(execution_protocol=execution_rec,method=method)
    monkeypatch.setattr(renderer,'checked',lambda r:tmp_path/r['path'])
    monkeypatch.setattr(renderer.aggregate,'validate_freeze',lambda *a:dict(execution_protocol=execution_rec,method=method))
    execution=dict(analysis_output='analysis');objects=[dict(index=i) for i in range(16,48)]
    monkeypatch.setattr(renderer.aggregate,'check_execution',lambda *a:(execution,{},{}))
    monkeypatch.setattr(renderer.aggregate,'eligible',lambda *a,**k:objects)
    monkeypatch.setattr(renderer.aggregate,'load_results',lambda *a:(list(values().values()),result_records))
    path=tmp_path/'analysis/reserved-execution.json';path.parent.mkdir()
    status=dict(terminal=False,completed=False,role='RESERVED_EVALUATION',method_sha256='m',execution_protocol_sha256='e',planned_object_indices=list(range(16,48)),
        completed_object_indices=list(range(16,48)),unlaunched_object_indices=[],failures=[])
    renderer.write(path,status)
    with pytest.raises(ValueError,match='terminal without failures'):renderer.gate_primary(recipe,dict(path='evaluation.json',sha256='unusedmock'))
    status.update(terminal=True,completed=True);renderer.write(path,status)
    returned=renderer.gate_primary(recipe,dict(path='evaluation.json',sha256='unusedmock'))
    assert sorted(returned[-1])==list(range(16,48))
    evaluation['reserved_results']=result_records[:-1];renderer.write(tmp_path/'evaluation.json',evaluation)
    with pytest.raises(ValueError,match='All32 exact sealed'):renderer.gate_primary(recipe,dict(path='evaluation.json',sha256='unusedmock'))


def test_recipe_fixed_forecast_reset_windows_and_source_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(renderer,'ROOT',tmp_path)
    for name in renderer.SOURCE_NAMES:
        p=tmp_path/'experiments'/name;p.parent.mkdir(exist_ok=True);p.write_text('synthetic frozen source')
    rec=renderer.prepare_recipe('recipe.json',dict(path='exec.json',sha256='e'),dict(path='method.json',sha256='m'),'rendered')
    recipe=renderer.validate_recipe(rec['path'],rec['sha256']);choices=recipe['choices']
    assert choices['A_frame_indices']==list(range(31,61)) and choices['B_frame_indices']==list(range(61))
    assert choices['motion_magnification']==1. and choices['fps']==20
    assert 'No observations during' in renderer.panel_labels('A')[1] and 'no B observations' in renderer.panel_labels('B')[1]
    with pytest.raises(ValueError,match='retained'):renderer.prepare_recipe('recipe.json',{}, {},'rendered')
    (tmp_path/'experiments'/renderer.SOURCE_NAMES[0]).write_text('changed')
    with pytest.raises(ValueError,match='source closure'):renderer.validate_recipe(rec['path'],rec['sha256'])


def mesh():
    return np.array([[0.,0.,0.],[.2,0.,0.],[0.,.1,0.],[0.,0.,.1]]),np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])


def test_synthetic_encoded_video_common_true_scale_orbit_no_interpolation(tmp_path):
    rest,faces=mesh();times=np.array([1.55,1.6]);actual=np.repeat(rest[None],2,axis=0);actual[:,1,2]=[.002,.003]
    means=[rest+.8*(actual-rest),rest+.9*(actual-rest)]
    r=renderer.render_sequence(rest,faces,times,[actual,*means],tmp_path/'synthetic.mp4',program='A',caption='SYNTHETIC ONLY')
    assert r['frames']==2 and r['bytes']>1000 and r['same_camera_scale_colors'] and r['motion_magnification']==1.
    assert r['first_simulation_time_s']==1.55 and r['last_simulation_time_s']==1.6
    assert r['maximum_vertices_outside_fixed_bounds']==[0,0,0]
    assert (tmp_path/'synthetic.first-frame.png').exists()
    with pytest.raises(ValueError,match='Fresh LOCAL'):renderer.render_sequence(rest,faces,times,[actual,*means],tmp_path/'synthetic.mp4',program='A')


def test_clipped_geometry_fails_preserves_evidence_without_autozoom(tmp_path):
    rest,faces=mesh();times=np.array([0.,.05]);q=np.repeat(rest[None],2,axis=0);q[:,1,2]=.5
    with pytest.raises(ValueError,match='Clipped geometry'):renderer.render_sequence(rest,faces,times,[q,q,q],tmp_path/'clipped.mp4',program='B')
    assert not (tmp_path/'clipped.mp4').exists()
    failure=renderer.read(tmp_path/'clipped.failure.json');assert failure['maximum_vertices_outside_fixed_bounds']==[1,1,1]
