"""Response-prior mechanics/causality contracts, static TRAIN geometry only."""
from copy import deepcopy
import pytest
import torch
from test_campaign_hybrid_predictor import static_case,inputs
from experiments.campaign_hybrid_response import HybridGaussianResponse,causal_history_basis
from experiments.campaign_hybrid_predictor import prepare_features


def network(data,tets,faces):
    torch.manual_seed(8103)
    return HybridGaussianResponse(data['rest'][0],tets,faces,width=8,layers=1).to(device=data['rest'].device,dtype=data['rest'].dtype)


def test_history_impulse_double_integral_and_fixed_clock():
    times=torch.arange(5,dtype=torch.float64)*.05
    controls=torch.zeros(1,5,12,dtype=torch.float64);support=torch.zeros(1,5,3,dtype=torch.float64)
    controls[0,:,0]=.1
    result=causal_history_basis(controls,support,times).reshape(1,5,15,14)
    torch.testing.assert_close(result[0,:,0,0],times,atol=1e-16,rtol=0)
    torch.testing.assert_close(result[0,:,0,1],times.square()/2,atol=1e-16,rtol=0)
    with pytest.raises(ValueError,match='fixed .05'):
        causal_history_basis(controls,support,times*2)


def test_sign_superposition_latents_and_zero_initial(static_case):
    data,tets,faces=inputs(static_case);model=network(data,tets,faces)
    a=deepcopy(data);b=deepcopy(data)
    a['controls'][:,1:,0]=.04;a['controls'][:,2:,7]=.02
    b['controls'][:,1:,2]=-.02;b['support_offsets'][:,1:,1]=.0003
    both=deepcopy(data);negative=deepcopy(a)
    for key in ['controls','support_offsets']:
        both[key]=a[key]+b[key];negative[key]=-a[key]
    ra,rb,rab,rminus=[model(x) for x in [a,b,both,negative]]
    for key in ['latent_displacements_m','handle_translations_m','handle_rotation_vectors_rad']:
        torch.testing.assert_close(rab[key],ra[key]+rb[key],atol=2e-16,rtol=2e-12)
        torch.testing.assert_close(rminus[key],-ra[key],atol=2e-16,rtol=2e-12)
        assert torch.equal(ra[key][:,0],torch.zeros_like(ra[key][:,0]))
    # Final nodal coordinates have exact fixtures; rigid patches use finite
    # rotations and are explicitly outside the superposition claim.
    zero=model(data);torch.testing.assert_close(zero['positions_m'],data['rest'][:,None].expand_as(zero['positions_m']),atol=1e-14,rtol=0)
    fixture=data['fixture_mask'][0]
    expected=data['rest'][:,None]+b['support_offsets'][:,:,None]
    torch.testing.assert_close(rb['positions_m'][:,:,fixture],expected[:,:,fixture],atol=1e-14,rtol=0)
    for j in range(2):
        ids=data['handle_patches'][0,j];local=data['rest'][0,ids]-data['handle_centers'][0,j]
        mapped=local@rab['handle_rotations'][0,:,j].transpose(-1,-2)+rab['handle_centers_m'][0,:,j,None]
        torch.testing.assert_close(rab['positions_m'][0,:,ids],mapped,atol=1e-14,rtol=0)


def test_causal_prefix_time_selection_and_prepared_features(static_case):
    data,tets,faces=inputs(static_case,frames=8);model=network(data,tets,faces)
    data['controls'][:,1:4,0]=.03
    full=model(data)
    changed=deepcopy(data);changed['controls'][:,4:]=torch.randn_like(changed['controls'][:,4:])
    changed['support_offsets'][:,4:]=.001
    torch.testing.assert_close(model(changed)['positions_m'][:,:4],full['positions_m'][:,:4],atol=1e-14,rtol=0)
    prefix={k:(v[:,:4] if k in ['controls','support_offsets'] else v[:4] if k=='times' else v) for k,v in data.items()}
    torch.testing.assert_close(model(prefix)['positions_m'],full['positions_m'][:,:4],atol=1e-14,rtol=0)
    selected=torch.tensor([0,2,7]);prepared=prepare_features(data,tets,faces)
    torch.testing.assert_close(model(data,time_indices=selected,prepared=prepared)['positions_m'],full['positions_m'][:,selected],atol=1e-14,rtol=0)
    data['controls'][:,0]=1e3
    torch.testing.assert_close(model(data)['positions_m'],full['positions_m'],atol=1e-14,rtol=0)
    assert not any('memory' in key for key in model.state_dict())


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:2',marks=pytest.mark.skipif(torch.cuda.device_count()<3,reason='CUDA2 unavailable'))])
def test_response_static_hypothesis_control_and_parameter_gradients(static_case,device):
    data,tets,faces=inputs(static_case,dtype=torch.float32,device=device);model=network(data,tets,faces)
    data['controls'][:,1:,0]=.04;data['controls'].requires_grad_();data['material'].requires_grad_()
    result=model(data)
    loss=(result['positions_m']-data['rest'][:,None]).square().sum()+result['handle_rotation_vectors_rad'].square().sum()
    loss.backward()
    for key in ['controls','material']:
        grad=data[key].grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().sum()>0)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
