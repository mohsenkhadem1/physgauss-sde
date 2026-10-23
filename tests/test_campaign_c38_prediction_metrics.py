import copy
import numpy as np
import pytest
from experiments import campaign_c38_prediction_metrics as m


def fixture():
    q = m.POINTS.copy(); q[:,2] += .001
    truth = dict(object_id='0'*32,private_configuration_index=0,repeat_index=0,
                 young_Pa=100000.,support_category=m.CATEGORIES[0],canonical_target_Fz_N=1.,
                 canonical_target_point_ids=np.arange(63),canonical_target_positions_m=q)
    public = dict(object_id='0'*32,observed_positions_m=q[m.POINT_IDS][None])
    interval = np.stack([q[m.POINT_IDS]-.0001,q[m.POINT_IDS],q[m.POINT_IDS]+.0001],-1)
    row = dict(completed=True,canonical_mean_positions_m=q,category_probabilities=[.8,.2],
               material_equal_tail_Pa=[90000.,100000.,110000.],
               material_quantile_details=[dict(unit_bracket=dict(lower=np.log(90000/75000)/np.log(2),upper=.3)),{},dict(unit_bracket=dict(lower=.4,upper=np.log(110000/75000)/np.log(2)))],
               canonical_position_equal_tail_m=interval,
               canonical_position_interval95_numerical_enclosure_m=interval[:,:,[0,2]],
               canonical_coordinate_variance_m2=np.ones((63,3))*1e-9,
               canonical_variance_integration_resolution_estimate_m2=np.ones((63,3))*1e-15)
    pred = dict(object_id=public['object_id'],completed=True,numerical_controls_passed=True,methods={'adaptive':row})
    return pred,truth,public


def test_exact_prediction_and_categorical_proper_scores():
    p,t,b=fixture(); r=m.score_prediction(p,t,b)
    assert r['canonical63']['coordinate_RMSE_m']==0
    assert r['material']['covered95'] and r['material']['absolute_log_median_error']==0
    assert r['canonical_pointwise95']['covered_count']==27
    assert r['support']['categorical_Brier']==pytest.approx(.08)
    assert r['support']['negative_log_probability']==pytest.approx(-np.log(.8))


def test_no_interval_widening_or_clipping_for_unconfident_errors():
    p,t,b=fixture(); row=p['methods']['adaptive']; q=t['canonical_target_positions_m'][m.POINT_IDS]
    row['canonical_position_equal_tail_m']=np.repeat((q+1e-9)[:,:,None],3,axis=2)
    row['canonical_position_interval95_numerical_enclosure_m']=np.stack([q-1e-8,q+1e-8],-1)
    row['category_probabilities']=[0.,1.]
    r=m.score_prediction(p,t,b)
    assert r['canonical_pointwise95']['covered_count']==0
    assert r['canonical_pointwise95']['maximum_miss_m']==pytest.approx(1e-9)
    assert r['canonical_pointwise95_numerical_quantile_enclosure']['covered_count']==27
    assert r['support']['negative_log_probability'] is None
    assert r['support']['infinite_negative_log_probability']


def test_private_probe_outcomes_rejected_and_identity_bound():
    p,t,b=fixture(); t['candidate_probe_observations_m']=np.zeros((2,9,3))
    with pytest.raises(ValueError,match='no probe'):m.score_prediction(p,t,b)
    del t['candidate_probe_observations_m']; b['object_id']='1'*32
    with pytest.raises(ValueError,match='identity'):m.score_prediction(p,t,b)


def test_numerical_failures_remain_in_all64_denominators():
    p,t,b=fixture(); rows=[]
    for c in range(4):
        for repeat in range(16):
            pred,truth,public=copy.deepcopy((p,t,b)); token=f'{c*16+repeat:032x}'
            pred['object_id']=truth['object_id']=public['object_id']=token
            truth.update(private_configuration_index=c,repeat_index=repeat)
            if repeat==0:pred['numerical_controls_passed']=False
            if repeat==1:pred['methods']={}
            rows.append(m.score_prediction(pred,truth,public))
    a=m.aggregate(rows)
    assert a['record_count']==64 and a['available_predictions']==60
    assert a['numerically_admitted']==56
    assert all(g['material95_covered_count']==14 and g['denominator_including_failures']==16 for g in a['groups'])
    with pytest.raises(ValueError,match='All64'):m.aggregate(rows[:-1])
