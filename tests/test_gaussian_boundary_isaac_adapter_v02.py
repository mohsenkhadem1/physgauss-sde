import pytest
import torch
from scripts.gaussian_boundary_flow_v02.isaac_flow_adapter import IsaacConditionalFlowAdapter


def test_teacher_schema_reproducibility_and_category_constraints():
    torch.set_num_threads(2);torch.manual_seed(83)
    model=IsaacConditionalFlowAdapter(width=32);x=torch.randn(2,5,119)
    def draw():return model.sample(x,samples=3,steps=2,generator=torch.Generator().manual_seed(44))
    a,b=draw(),draw()
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert a['effective_physical'].shape==(2,3,25) and a['boundary_type'].shape==(2,3)
    p=a['effective_physical'];bc=a['boundary_type'];f=a['force_type']
    assert torch.all(p[...,22:25][bc!=3]==0)
    assert torch.allclose(p[...,19:22][f==3],-p[...,16:19][f==3])
    assert torch.all(p[...,19:22][f!=3]==0)
    assert torch.allclose(p[...,7:10][bc!=2],p[...,4:7][bc!=2])


def test_training_gradient_and_future_feature_rejection():
    model=IsaacConditionalFlowAdapter(width=32)
    physical=model.transform.unconstrained_to_physical(torch.zeros(2,25))
    loss=model.loss(torch.zeros(2,5,119),physical,torch.tensor([0,1]),torch.tensor([2,3]));loss.backward()
    assert torch.isfinite(loss) and model.flow.vector[-1].weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='Exactly five'):model.sample(torch.zeros(2,10,119))


@pytest.mark.parametrize('kind',['flow','mdn'])
def test_logit_standardization_persisted_and_guarded(kind):
    from scripts.gaussian_boundary_flow_v02.isaac_flow_adapter import IsaacDiagonalMixtureAdapter
    cls=IsaacConditionalFlowAdapter if kind=='flow' else IsaacDiagonalMixtureAdapter
    model=cls(width=32);mean=torch.linspace(-1,1,25);scale=torch.linspace(.1,2,25)
    model.set_standardization(mean,scale)
    restored=cls(width=32);restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.logit_mean,mean) and torch.equal(restored.logit_scale,scale)
    with pytest.raises(ValueError,match='positive scale'):model.set_standardization(mean,torch.zeros(25))
    with pytest.raises(ValueError,match='integration steps'):model.sample(torch.zeros(1,5,119),steps=0)
    p=model.transform.unconstrained_to_physical(torch.zeros(1,25))
    with pytest.raises(ValueError,match='category weight'):model.loss(torch.zeros(1,5,119),p,torch.tensor([0]),torch.tensor([0]),category_weight=float('nan'))


def test_mixture_density_standard_gaussian_and_gradient():
    import math
    from scripts.gaussian_boundary_flow_v02.isaac_flow_adapter import IsaacDiagonalMixtureAdapter
    model=IsaacDiagonalMixtureAdapter(width=32)
    # All four equal components are N(0,I): exact standardized per-coordinateNLL.
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    x=torch.zeros(2,5,119);physical=model.transform.unconstrained_to_physical(torch.zeros(2,25))
    loss=model.loss(x,physical,torch.tensor([0,1]),torch.tensor([2,3]))
    assert float(loss.detach())==pytest.approx(.5*math.log(2*math.pi)+6*math.log(4),abs=1e-5)
    loss.backward();assert model.mixture[-1].bias.grad is not None and torch.isfinite(model.mixture[-1].bias.grad).all()
    a=model.sample(x,samples=3,generator=torch.Generator().manual_seed(4))
    b=model.sample(x,samples=3,generator=torch.Generator().manual_seed(4))
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_batch_samples_decode_per_case_teacher_contract():
    import numpy as np
    from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import predict_particles
    model=IsaacConditionalFlowAdapter(width=32)
    draw=model.sample(torch.zeros(2,5,119),samples=3,steps=2,generator=torch.Generator().manual_seed(9))
    calls=[]
    def teacher(rest,times,physical,boundary,force):
        assert physical.ndim==2 and boundary.ndim==force.ndim==1
        calls.append(len(physical))
        return rest[None,None].expand(len(physical),len(times),-1,-1)+physical[:,None,None,:3]*0
    for case in range(2):
        per_case={k:v[case] for k,v in draw.items()}
        prediction=predict_particles(dict(device='cpu',forward=teacher),per_case,dict(rest_points_m=np.zeros((7,3),np.float32)),[0,1],chunk_size=2)
        assert prediction.shape==(3,2,7,3)
    assert calls==[2,1,2,1]
