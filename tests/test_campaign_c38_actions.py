import hashlib
import inspect
import numpy as np
import pytest
from experiments.campaign_c38_actions import select_reset_actions,validate_reset_actions,freeze_choices,update_from_frozen


def actions(n):
    return [dict(id=f'action-{i}',kind='reset_probe',duration_s=.2,frame_count=1,sensor_dimension=1,reset_to_common_rest=True,common_state_sha256='0'*64,target_history_sha256='1'*64,force_domain_sha256='2'*64) for i in range(n)]


def gauss(n):
    x,w=np.polynomial.hermite.hermgauss(n)
    return x*np.sqrt(2),w/np.sqrt(np.pi)


def test_gaussian_limit_risk_information_and_update(tmp_path):
    x,w=gauss(51);p=x[None,:,None];cov=np.ones((1,1));a=actions(1)
    result=select_reset_actions(x[:,None],p,cov,w,a,object_index=0,fixed_index=0,fantasies=4096)
    assert abs(result['expected_target_variance_m2'][0]-.5)<1e-7
    assert abs(result['expected_information_gain_nats'][0]-.5*np.log(2))<.03
    source=tmp_path/'source.py';source.write_text('frozen synthetic source')
    path=tmp_path/'choice.json';digest=freeze_choices(path,result,{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    post,_=update_from_frozen(path,digest,'fixed',p,cov,w,np.array([.4]))
    assert abs(post@x-.2)<1e-8
    assert abs(post@np.square(x-.2)-.5)<1e-8
    with pytest.raises(FileExistsError):freeze_choices(path,result,{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='base posterior'):update_from_frozen(path,digest,'fixed',p,cov,post,np.array([.4]))


def test_information_can_prefer_target_irrelevant_sensor():
    x,w=gauss(21);xx,yy=np.meshgrid(x,x,indexing='ij');prior=np.outer(w,w).ravel()
    target=xx.reshape(-1,1);pred=np.stack([xx.ravel(),3*yy.ravel()])[:,:,None]
    r=select_reset_actions(target,pred,np.eye(1),prior,actions(2),object_index=4,fixed_index=0,fantasies=256)
    assert r['policies']['predictive_risk']==0
    assert r['policies']['information_gain']==1
    assert np.isclose(r['expected_target_variance_m2'][1],1,atol=1e-12)
    assert 'observation' not in inspect.signature(select_reset_actions).parameters
    assert r==select_reset_actions(target,pred,np.eye(1),prior,actions(2),object_index=4,fixed_index=0,fantasies=256)


def test_budget_target_reset_and_hash_barriers(tmp_path):
    a=actions(2);a[1]['target_history_sha256']='3'*64
    with pytest.raises(ValueError,match='canonical target'):validate_reset_actions(a,np.eye(1))
    a=actions(2);a[1]['duration_s']=.3
    with pytest.raises(ValueError,match='budgets'):validate_reset_actions(a,np.eye(1))
    a=actions(2);a[1]['reset_to_common_rest']=False
    with pytest.raises(ValueError,match='reset'):validate_reset_actions(a,np.eye(1))
    x,w=gauss(7);p=x[None,:,None];r=select_reset_actions(x[:,None],p,np.eye(1),w,actions(1),object_index=0,fixed_index=0)
    source=tmp_path/'source.py';source.write_text('a');path=tmp_path/'choice.json'
    digest=freeze_choices(path,r,{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='model'):update_from_frozen(path,digest,'fixed',p*2,np.eye(1),w,np.array([0.]))
    source.write_text('b')
    with pytest.raises(ValueError,match='source'):update_from_frozen(path,digest,'fixed',p,np.eye(1),w,np.array([0.]))
