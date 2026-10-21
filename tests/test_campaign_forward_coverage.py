import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_forward_coverage import (ResponseInterpolator,leave_one_layout_out,
    load_selection_metadata,stratum,descriptor,score)


def record(name,x,value,E=5000.,bc=0,action='poke_0',split='train'):
    rest=np.array([[0.,0.,0.],[1.,1.,1.]])
    physical=np.zeros(25,dtype=np.float64)
    physical[:4]=[np.log(E),np.log(600),.35,np.log(.5)]
    physical[4:10]=[x,.5,0,x,.5,0]
    physical[10:16]=[0,.2,.5,1,.8,.5]
    axis=min(int(action[-1]),2);physical[16+axis]=.02
    q=np.broadcast_to(rest,(2,*rest.shape)).copy();q[1,:,0]+=value
    return {'episode_id':name+'/'+action,'group_id':name,'physical':physical,
        'bc':bc,'force':2 if action=='poke_3' else 0,'rest':rest,'q':q,
        'times':np.array([0.,3.]),'split':split}


def test_nearest_uses_known_descriptor_never_query_response():
    a,b=record('t0',.1,1),record('t1',.9,3)
    model=ResponseInterpolator('nearest_raw').fit([a,b])
    query=record('v0',.11,999,split='validation')
    np.testing.assert_allclose(model.predict(query),a['q'])
    query['q'][:]=np.nan
    np.testing.assert_allclose(model.predict(query),a['q'])


def test_stationary_force_over_E_scaling_and_moving_exception():
    a=record('t0',.1,2e-5,E=2000.)
    query=record('v0',.1,0,E=8000.,split='validation')
    expected=query['rest'].copy();expected[:,0]+=5e-6
    np.testing.assert_allclose(ResponseInterpolator('nearest_fe').fit([a]).predict(query)[1],expected,atol=1e-15)
    a['bc']=query['bc']=3
    np.testing.assert_allclose(ResponseInterpolator('nearest_fe').fit([a]).predict(query),a['q'])


def test_fixed_rbf_far_from_train_returns_response_mean_without_clipping():
    a,b=record('t0',.1,1),record('t1',.9,3)
    query=record('v0',1000.,0,split='validation')
    model=ResponseInterpolator('rbf_raw').fit([a,b])
    np.testing.assert_allclose(model.predict(query),.5*(a['q']+b['q']),atol=1e-14)
    coverage=model.coverage(query)
    assert any(coverage['outside_train_coordinate_range'])
    assert coverage['train_stratum_count']==2


def test_all_actions_of_layout_are_left_out_and_preprocessing_refitted():
    records=[record('t0',.1,1,action=a) for a in ['poke_0','poke_1']]+[
             record('t1',.9,3,action=a) for a in ['poke_0','poke_1']]
    predictions=leave_one_layout_out(records,'rbf_raw')
    np.testing.assert_allclose(predictions[0],records[2]['q'],atol=1e-7)
    np.testing.assert_allclose(predictions[1],records[3]['q'],atol=1e-7)
    np.testing.assert_allclose(predictions[2],records[0]['q'],atol=1e-7)


def test_grouped_boundary_direction_family_cannot_borrow_wrong_stratum():
    a,b=record('t0',.1,1),record('t1',.1,3,action='poke_3')
    model=ResponseInterpolator('nearest_raw').fit([a,b])
    np.testing.assert_allclose(model.predict(record('v0',.1,0,action='poke_3')),b['q'])
    with pytest.raises(ValueError):model.predict(record('v0',.1,0,bc=1))
    invalid=record('v0',.1,0);invalid['force']=2
    with pytest.raises(ValueError):stratum(invalid)


def test_fitting_validation_rejected_before_q_access():
    query=record('v0',.1,0,split='validation');del query['q']
    with pytest.raises(ValueError,match='TRAIN'):ResponseInterpolator('rbf_raw').fit([query])


def test_gate_or_calibration_role_rejected_before_any_dataset_open():
    config={'new_validation_roles':{'selection':['v0000']},
            'expanded_validation_pool':'DOES-NOT-EXIST',
            'dataset':'DOES-NOT-EXIST'}
    for prohibited in ['v0008','v0015','t0112','h0000']:
        with pytest.raises(ValueError,match='authorized'):load_selection_metadata(config,['v0000',prohibited])


def test_descriptor_contract_and_coordinate_scores():
    r=record('t0',.1,1)
    assert descriptor(r).shape==(19,)
    result=score(np.stack([r['q']]),[r])
    assert result['aggregate']['coordinate_rmse_m']==0
    assert result['aggregate']['static_coordinate_rmse_m']>0
