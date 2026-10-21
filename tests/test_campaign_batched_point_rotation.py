import numpy as np
import pytest
import torch
from experiments import campaign_batched_point_rotation as c
from experiments.campaign_point_rotation_retraction import retract_point_rotation


def test_cpu_matches_numpy_for_distinct_candidate_pins_and_keeps_elastic_residual():
    X,q,pins=c.synthetic();got=c.retract_positions(torch.tensor(X),torch.tensor(q),torch.tensor(pins))
    for b,pin in enumerate(pins):
        ref=retract_point_rotation(X,q[b]-X,int(pin))
        np.testing.assert_allclose(got['correction_m'][b],ref['correction_m'],atol=1e-16,rtol=1e-12)
        np.testing.assert_allclose(got['positions_m'][b],X+ref['displacement_m'],atol=1e-16,rtol=1e-12)
        np.testing.assert_array_equal(got['positions_m'][b,:,pin],np.broadcast_to(X[pin],(7,3)))
        r=X-X[pin]
        rigid=np.einsum('tij,nj->tni',got['fitted_rotation_matrix'][b].numpy(),r)-r
        np.testing.assert_allclose(got['positions_m'][b].numpy()-X-rigid,.002*r[None]+np.zeros_like(q[b]),atol=2e-16,rtol=1e-11)


def test_pure_firstorder_rotation_becomes_exact_SO3_without_handle_claim():
    X,q,pins=c.synthetic();r=X-X[0];omega=torch.tensor([.1,-.2,.05],dtype=torch.float64)
    positions=torch.tensor(X)+torch.linalg.cross(omega.expand(len(r),3),torch.tensor(r))
    got=c.retract_positions(torch.tensor(X),positions[None,None],torch.tensor([0]))
    wx,wy,wz=omega;skew=torch.tensor([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]],dtype=torch.float64)
    R=torch.matrix_exp(skew)
    torch.testing.assert_close(got['positions_m'][0,0],torch.tensor(r)@R.T+torch.tensor(X[0]),atol=1e-16,rtol=1e-13)
    assert 'no improved dynamics' in got['role']


def test_passive_frame_equivariance_with_public_rest_and_candidate_pins():
    X,q,pins=c.synthetic();a=.73;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    t=np.array([.3,-.6,.4]);first=c.retract_positions(torch.tensor(X),torch.tensor(q),torch.tensor(pins))
    other=c.retract_positions(torch.tensor(X@Q.T+t),torch.tensor(q@Q.T+t),torch.tensor(pins))
    np.testing.assert_allclose(other['positions_m'],first['positions_m'].numpy()@Q.T+t,atol=4e-16,rtol=1e-13)


def test_zero_angle_autograd_is_finite_and_identity_to_first_order():
    X,_,_=c.synthetic();rest=torch.tensor(X);q=rest[None,None].clone().requires_grad_()
    pins=torch.tensor([0]);result=c.retract_positions(rest,q,pins)
    weight=torch.arange(q.numel(),dtype=torch.float64).reshape_as(q)
    # Only outputs in admissible free-node directions are differentiated.
    gradient=torch.autograd.grad((result['positions_m']*weight).sum(),q)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient,weight,atol=0,rtol=0)
    assert torch.count_nonzero(result['correction_m'])==0


def test_drift_and_singular_normal_matrix_rejected_without_repair():
    X,q,pins=c.synthetic();q[0,0,pins[0],0]+=1e-15
    with pytest.raises(ValueError,match='exact'):c.retract_positions(torch.tensor(X),torch.tensor(q),torch.tensor(pins))
    line=torch.tensor([[0.,0,0],[1.,0,0],[2.,0,0]],dtype=torch.float64)
    with pytest.raises(ValueError,match='normal matrix'):c.retract_positions(line,line[None,None],torch.tensor([0]))
