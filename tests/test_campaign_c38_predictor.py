import numpy as np
import torch
from experiments.campaign_static_p2_block import enrich
from experiments.campaign_c38_predictor import BlockGaussianPredictor,sparse_adjacency,exchange

def setup():
    x=np.array([[0.,0,0],[.2,0,0],[0,.1,0],[0,0,.08]])
    X,E=enrich(x,np.array([[0,1,2,3]]));X=torch.tensor(X,dtype=torch.float64);E=torch.tensor(E)
    faces=torch.tensor([[0,4,5],[4,1,7],[5,7,2],[4,7,5]])
    model=BlockGaussianPredictor(X,E,faces,[1.,1.],width=8,layers=1).double()
    mask=torch.zeros(1,len(X),dtype=torch.bool);mask[:,0]=True
    context=dict(rest=X[None],material=torch.tensor([[5000.,625.,.35,0.]],dtype=X.dtype),fixture_mask=mask,load_basis_N=torch.zeros(1,2,len(X),3,dtype=X.dtype),
       load_centers_material_m=torch.tensor([[[.15,0,0],[.05,0,0]]],dtype=X.dtype),load_covariances_material_m2=torch.eye(3,dtype=X.dtype)[None,None].repeat(1,2,1,1)*.02**2,load_active=torch.tensor([[True,False]]),information_role='known_physics_oracle_A')
    context['load_basis_N'][0,0,1,0]=1;context['load_basis_N'][0,1,2,1]=1
    return model,context,E

def test_sparse_exchange_matches_dense_normalized_operator_and_gradient():
    model,data,E=setup();a=model.adjacency;h=torch.randn(2,len(data['rest'][0]),5,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(exchange(a,h),a.to_dense()@h)
    exchange(a,h).square().sum().backward();assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    assert a.layout==torch.sparse_coo and a._nnz()<=len(data['rest'][0])**2

def test_zero_drive_exact_rest_and_fixed_moving_targets():
    model,data,_=setup();times=torch.arange(5,dtype=torch.float64)*.05;coeff=torch.zeros(1,5,2,dtype=torch.float64)
    q=model(data,coeff,times)['positions_m'];torch.testing.assert_close(q,data['rest'][:,None].expand(-1,5,-1,-1),rtol=0,atol=0)
    offsets=torch.zeros(1,5,3,dtype=torch.float64);offsets[0,:,2]=times*.01
    q=model(data,coeff,times,support_offsets=offsets)['positions_m']
    torch.testing.assert_close(q[:,:,0],data['rest'][:,None,0]+offsets,rtol=0,atol=0)

def test_future_control_and_unused_truth_cannot_change_past_predictions():
    model,data,_=setup();times=torch.arange(6,dtype=torch.float64)*.05;coeff=torch.randn(1,6,2,dtype=torch.float64)
    first=model(data,coeff,times)['positions_m'];changed=coeff.clone();changed[:,4:]+=100
    data['unavailable_future_observations']=torch.full((1,100,3),float('nan'))
    other=model(data,changed,times)['positions_m']
    torch.testing.assert_close(first[:,:4],other[:,:4],rtol=0,atol=0)
    torch.testing.assert_close(first[:,:1],data['rest'][:,None],rtol=0,atol=0)

def test_known_candidate_role_zero_drag_and_training_gradients():
    model,data,_=setup();data['information_role']='candidate_physics_B';times=torch.arange(4,dtype=torch.float64)*.05;coeff=torch.ones(1,4,2,dtype=torch.float64,requires_grad=True)
    result=model(data,coeff,times,time_indices=torch.tensor([1,3]));assert result['positions_m'].shape==(1,2,len(data['rest'][0]),3)
    result['positions_m'].square().sum().backward()
    assert torch.isfinite(coeff.grad).all() and coeff.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
