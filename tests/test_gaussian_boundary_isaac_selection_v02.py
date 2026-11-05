import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import isaac_validation_selection as selection


def make_rows():
    metrics={k:1. for k in ['coordinate_rmse_m','rigid_aligned_shape_rmse_m','coordinate90_coverage','coordinate90_width_m','coordinate90_interval_score_m','trajectory_energy_score_m','active_parameter90_coverage','active_parameter90_normalized_width','active_parameter_normalized_absolute_error']}
    return [dict(case=f'v{i}',candidate=name,state='complete',metrics=dict(metrics)) for i in range(44) for name in selection.CANDIDATES]


def test_only_recorded_coordinate_score_selects_and_exact_tie_keeps_best():
    rows=make_rows()
    for r in rows:
        if r['candidate']=='flow_last':r['metrics']['coordinate_rmse_m']=.8;r['metrics']['rigid_aligned_shape_rmse_m']=10.
    result=selection.choose(rows)
    assert result['choices']['flow']['selected']=='flow_last'
    assert result['choices']['mdn']['selected']=='mdn_best'
    assert result['test_re_evaluation'] is False


def test_any_failure_makes_candidate_ineligible_without_case_dropping():
    rows=make_rows()
    for r in rows:
        if r['candidate']=='flow_last':r['metrics']['coordinate_rmse_m']=.01
    row=next(r for r in rows if r['candidate']=='flow_last');row['state']='prediction_failed'
    result=selection.choose(rows)
    assert not result['candidates']['flow_last']['eligible'] and result['choices']['flow']['selected']=='flow_best'
    assert result['candidates']['flow_last']['complete']==43 and result['candidates']['flow_last']['failed']==1


def test_test_role_rejected_before_trajectory_access(tmp_path,monkeypatch):
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(cases=[dict(key='test',split='test',path=str(tmp_path/'episode'))])))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('Forbidden target opened'))
    with pytest.raises(ValueError,match='VALIDATION'):
        selection.validation_truth(dict(key='test',episode=str(tmp_path/'episode')),dict(catalog=str(catalog)),{})


def test_absent_barrier_rejects_before_validation_trajectory_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,'verify',lambda *a:{})
    monkeypatch.setattr(selection,'validation_truth',lambda *a:pytest.fail('Target read before barrier'))
    with pytest.raises(ValueError,match='barrier absent'):selection.score(tmp_path)


def test_barrier_requires_all176_terminal_records(tmp_path,monkeypatch):
    plan=dict(cases=[dict(case=f'v{i}') for i in range(44)]);(tmp_path/'plan.json').write_text('{}')
    monkeypatch.setattr(selection,'verify',lambda *a:plan)
    monkeypatch.setattr(selection,'terminal',lambda *a:None)
    with pytest.raises(ValueError,match='All176'):selection.barrier(tmp_path)
    assert not(tmp_path/'barrier.json').exists()


def metadata_roster(tmp_path):
    catalog=dict(cases=[dict(key=f'dataset/v{i}',split='validation',path=str(tmp_path/'dataset'/'episodes'/f'v{i}'),group_id=f'g{i}',recipe_sha256=str(i)) for i in range(44)])
    manifest=dict(cases=[dict(key=r['key'],role='validation',dataset='dataset',group=r['group_id'],recipe_sha256=r['recipe_sha256'],cache=f'{i:03d}.npz',cache_sha256=f'hash{i}') for i,r in enumerate(catalog['cases'])])
    test=dict(datasets=[dict(root=str(tmp_path/'dataset'),cases=[dict(episode_id='test',group_id='testgroup')])])
    return manifest,catalog,test


@pytest.mark.parametrize('mutation',['wrong_dataset','test_path','test_group'])
def test_malformed_validation_identity_rejected_before_targets(tmp_path,monkeypatch,mutation):
    manifest,catalog,test=metadata_roster(tmp_path)
    assert len(selection.bound_validation_roster(manifest,catalog,test))==44
    monkeypatch.setattr(selection,'validation_truth',lambda *a:pytest.fail('Target accessed while checking metadata'))
    if mutation=='wrong_dataset':manifest['cases'][0]['dataset']='relabeled'
    elif mutation=='test_path':catalog['cases'][0]['path']=str(tmp_path/'dataset'/'episodes'/'test')
    else:manifest['cases'][0]['group']='testgroup';catalog['cases'][0]['group_id']='testgroup'
    with pytest.raises(ValueError):selection.bound_validation_roster(manifest,catalog,test)
