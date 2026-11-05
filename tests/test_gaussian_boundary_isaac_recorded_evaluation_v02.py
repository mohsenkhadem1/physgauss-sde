import json
from pathlib import Path
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import isaac_recorded_evaluation as score


def test_rigid_alignment_removes_pose_but_preserves_shape_error():
    rng=np.random.default_rng(4);points=rng.normal(size=(2,20,3));angle=.6
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    transformed=points@rotation+np.array([2,-1,3])
    assert score.rigid_aligned_shape_rmse(transformed,points)<1e-12
    assert score.rigid_aligned_shape_rmse(2*transformed,points)>.5


def test_energy_and_interval_scores_have_known_values():
    target=np.zeros((2,3,3));rest=np.zeros((3,3));samples=np.stack([-np.ones_like(target),np.ones_like(target)])
    result=score.trajectory_metrics(samples,target,rest)
    assert result['coordinate_rmse_m']==0 and result['trajectory_energy_score_m']==pytest.approx(.5)
    assert result['coordinate90_coverage']==1 and result['coordinate90_width_m']==pytest.approx(1.8)
    deterministic=score.trajectory_metrics(np.ones((1,2,3,3)),target,rest)
    assert deterministic['trajectory_energy_score_m']==pytest.approx(1.)
    assert deterministic['coordinate90_interval_score_m']==pytest.approx(20.)
    assert deterministic['rigid_aligned_shape_rmse_m']==0


def identity_arrays():
    rest=np.arange(455*3,dtype=np.float32).reshape(455,3)/10000
    times=np.arange(61)/20;target=np.broadcast_to(rest,(61,455,3)).copy();positions=np.broadcast_to(rest,(2,61,455,3)).copy()
    public=dict(rest=rest,frame_ids=np.arange(5)*5,times=np.arange(5)*.25)
    arrays=dict(positions_m=positions,mean_positions_m=positions.mean(0),times_s=times.astype(np.float32),rest_points_m=rest.copy(),input_frame_ids=public['frame_ids'].copy(),input_times_s=public['times'].copy(),physical=np.zeros((2,25)),effective_physical=np.zeros((2,25)))
    return arrays,public,target,rest,times


@pytest.mark.parametrize('mutation',['clock','reference','initial_order','prefix','mean','sample_count'])
def test_identity_rejects_mismatch_without_interpolation(mutation):
    arrays,public,target,rest,times=identity_arrays()
    score.validate_identity(arrays,public,target,rest,times,times,2)
    if mutation=='clock':times=times+.001
    elif mutation=='reference':arrays['rest_points_m']=rest[::-1].copy()
    elif mutation=='initial_order':target[0]=target[0,::-1]
    elif mutation=='prefix':arrays['input_frame_ids'][-1]=30
    elif mutation=='mean':arrays['mean_positions_m']+=.01
    else:arrays['positions_m']=arrays['positions_m'][:1]
    with pytest.raises(ValueError):score.validate_identity(arrays,public,target,rest,times,times,2)


def test_absent_barrier_blocks_before_any_target_loader(tmp_path,monkeypatch):
    cohort=tmp_path/'cohort';cohort.mkdir();(cohort/'protocol.json').write_text('{}')
    output=tmp_path/'score';output.mkdir()
    plan=dict(cohort=str(cohort),protocol_sha256=score.sha(cohort/'protocol.json'),source_sha256={str(Path(score.__file__).resolve().relative_to(score.ROOT)):score.sha(score.__file__)})
    (output/'evaluation-plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(score,'load_truth_after_gate',lambda *a:pytest.fail('Targets opened before gate'))
    with pytest.raises(ValueError,match='barrier is absent'):score.run(output)


def test_active_metrics_do_not_reward_inactive_padding():
    active=score.active_coordinates(0,0)
    assert active.sum()==13 and not active[7:10].any() and not active[13:16].any() and not active[19:25].any()
    assert score.active_coordinates(2,3)[7:16].all()
    assert not score.active_coordinates(2,3)[19:22].any()
    assert score.active_coordinates(3,0)[22:25].all()


def test_summary_keeps_missing_failures_and_paired_denominators():
    cases=[dict(case=f'case{i}',dataset='neural' if i<64 else 'orbit',group=f'g{i}') for i in range(76)]
    metrics={k:1. for k in score.SUMMARY_METRICS};metrics['categories']={name:dict(correct=True,brier=0.) for name in ('boundary_type','force_type')}
    rows=[dict(cases[0],model='flow',state='complete',metrics=metrics),dict(cases[0],model='mdn',state='complete',metrics=dict(metrics,coordinate_rmse_m=2.)),dict(cases[1],model='flow',state='prediction_failed')]
    result=score.summarize(rows,dict(cases=cases));flow=result['methods']['flow']
    assert (flow['expected'],flow['complete'],flow['failed'],flow['missing'])==(76,1,1,74)
    pair=result['paired']['all']['mdn']
    assert pair['paired_cases']==1 and len(pair['excluded_cases'])==75
    assert pair['metrics']['coordinate_rmse_m']['episode_mean_flow_minus_baseline']==-1
    assert result['datasets']['orbit']['flow']['expected']==12


def test_reflection_is_not_removed_as_a_rigid_rotation():
    points=np.random.default_rng(4).normal(size=(3,30,3));reflected=points.copy();reflected[...,0]*=-1
    assert score.rigid_aligned_shape_rmse(reflected,points)>.5


@pytest.mark.parametrize('gate_error',['incomplete380','tampered prediction seal'])
def test_present_invalid_barrier_blocks_targets(tmp_path,monkeypatch,gate_error):
    cohort=tmp_path/'cohort';cohort.mkdir();(cohort/'protocol.json').write_text('{}');(cohort/'barrier.json').write_text('{}')
    output=tmp_path/'score';output.mkdir()
    plan=dict(cohort=str(cohort),protocol_sha256=score.sha(cohort/'protocol.json'),source_sha256={str(Path(score.__file__).resolve().relative_to(score.ROOT)):score.sha(score.__file__)})
    (output/'evaluation-plan.json').write_text(json.dumps(plan))
    def reject(*a):raise ValueError(gate_error)
    monkeypatch.setattr(score.prediction,'barrier',reject)
    monkeypatch.setattr(score,'load_truth_after_gate',lambda *a:pytest.fail('Target loader invoked through invalid barrier'))
    with pytest.raises(ValueError,match=gate_error):score.run(output)


def test_existing_score_rejects_changed_target_even_with_same_binding(tmp_path):
    row=dict(case='case',dataset='dataset',group='group');run=tmp_path/'case'/'flow';run.mkdir(parents=True)
    (run/'status.json').write_text(json.dumps(dict(state='complete')));(run/'prediction-seal.json').write_text('{}')
    target=tmp_path/'target.npz';target.write_bytes(b'original')
    binding=dict(protocol='same')
    record=dict(row,model='flow',state='complete',binding=binding,prediction_status_sha256=score.sha(run/'status.json'),prediction_seal_sha256=score.sha(run/'prediction-seal.json'),target_source_sha256={str(target):score.sha(target)})
    score.verify_existing_score(record,row,'flow',tmp_path/'case',binding)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='target binding'):score.verify_existing_score(record,row,'flow',tmp_path/'case',binding)
