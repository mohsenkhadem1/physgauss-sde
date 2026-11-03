import numpy as np
import torch
import pytest
from scripts.complex_scenes_v18.phystwin_autograd import spring_force


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_native_gradient_matches_direct_energy_force_derivative(device):
    x=torch.tensor([[0.,0.,0.],[.022,.003,0.],[.04,0.,.002]],requires_grad=True,device=device)
    v=torch.tensor([[.1,0.,0.],[0.,.2,0.],[0.,0.,-.1]],requires_grad=True,device=device)
    y=torch.tensor([np.log(.5),np.log(.7)],dtype=torch.float32,requires_grad=True,device=device)
    edges=torch.tensor([[0,1],[1,2]],dtype=torch.int32,device=device);rest=torch.tensor([.02,.02],device=device);weight=torch.arange(9,dtype=torch.float32,device=device).reshape(3,3)/9
    native=spring_force(x,v,y,edges,rest,.03);loss=(native*weight).sum()
    gradients=torch.autograd.grad(loss,(x,v,y))
    delta=x[edges[:,1].long()]-x[edges[:,0].long()];length=delta.norm(dim=-1);direction=delta/length[:,None]
    rel=((v[edges[:,1].long()]-v[edges[:,0].long()])*direction).sum(-1)
    f=(y.exp()*(length/rest-1)+.03*rel)[:,None]*direction
    expected=torch.zeros_like(x).index_add(0,edges[:,0].long(),f).index_add(0,edges[:,1].long(),-f)
    direct=torch.autograd.grad((expected*weight).sum(),(x,v,y))
    torch.testing.assert_close(native,expected,rtol=2e-5,atol=1e-6)
    for actual,want in zip(gradients,direct):torch.testing.assert_close(actual,want,rtol=1e-4,atol=2e-6)
    assert x.grad is None and v.grad is None and y.grad is None
