import numpy as np
import pytest
from scripts.benchmark_improvement_v14.independent_audit_v14 import paired,frame_scores
from scripts.benchmark_improvement_v14.summarize import compare


def row(name,a,b=None,valid=3):
    methods={'benchmark_v14':{'summary':dict(coordinate_rmse_mm=a,failed_frames=3-valid,valid_frames=valid,total_frames=3)}}
    if b is not None:methods['base']={'summary':dict(coordinate_rmse_mm=b,failed_frames=0,valid_frames=3,total_frames=3)}
    return dict(case=dict(name=name),metrics=methods)


def test_unequal_coverage_and_missing_baseline_do_not_mix_means():
    rows=[row('paired',2,5),row('missingbaseline',100),row('partial',4,8,2)]
    pairs=paired(rows,'base','coordinate_rmse_mm')
    assert pairs==[('paired',2.,5.),('partial',4.,8.)]
    got=compare(rows,'base','coordinate_rmse_mm')
    assert got['available_pairs']==2 and got['declared_cases']==3
    assert got['proposal_mean_mm']==3 and got['baseline_mean_mm']==6.5
    assert got['mean_improvement_mm']==3.5 and got['unavailable']==['missingbaseline']
    assert got['all_cases_and_frames_complete'] is False


def test_equal_frame_rmse_and_proper_shape_with_failed_frames():
    rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    truth=np.repeat(rest[None],3,axis=0);pred=truth.copy();pred[0]+=np.array([.003,0,0]);pred[1]+=np.array([.009,0,0]);pred[2]=np.nan
    got=frame_scores(pred,truth,0)
    assert got['coordinate_rmse_mm']==pytest.approx(6/np.sqrt(3))
    assert got['aligned_shape_rmse_mm']<1e-10
    assert (got['valid_frames'],got['failed_frames'],got['total_frames'])==(2,1,3)
    with pytest.raises(ValueError,match='Negative'):paired([row('bad',-1,2)],'base','coordinate_rmse_mm')


def test_complete_subset_zero_pairs_and_malformed_counts():
    partial=row('partial',4,8,2);full=row('full',2,5)
    got=compare([full,partial],'base','coordinate_rmse_mm')
    assert got['complete_pairs']['count']==1
    assert got['complete_pairs']['proposal_mean_mm']==2
    missing=compare([row('none',2)],'base','coordinate_rmse_mm')
    assert missing['declared_cases']==1 and missing['available_pairs']==0
    assert missing['metric']=='coordinate_rmse_mm' and not missing['all_cases_and_frames_complete']
    mismatch=row('counts',2,3);mismatch['metrics']['base']['summary'].update(total_frames=4,valid_frames=4)
    assert not compare([mismatch],'base','coordinate_rmse_mm')['all_cases_and_frames_complete']
    mismatch['metrics']['base']['summary']['failed_frames']=1
    with pytest.raises(ValueError,match='coverage'):compare([mismatch],'base','coordinate_rmse_mm')
