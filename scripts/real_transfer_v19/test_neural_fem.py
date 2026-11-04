import itertools
import numpy as np
import torch
from scripts.real_transfer_v19.neural_fem import NeuralFEM, parameters


def small_model():
    nodes=np.array(list(itertools.product(np.linspace(-.1,.1,3),repeat=3)))
    support=np.array(list(itertools.product(np.linspace(-.1,.1,9),repeat=3)))
    weights=np.exp(-((nodes[:,0]+.1)/.04)**2)[None]
    data=dict(nodes=nodes,support_geometry=support,attachment_weights=weights,
              controller_targets=nodes[None,None],attachment_active=np.ones((1,1)),floor=-.1)
    return NeuralFEM(data,device='cpu',materials=2)


def test_energy_residual_and_tangent():
    torch.set_num_threads(1);m=small_model();th=m.t(parameters(m)[0])
    x=(m.rest @ m.t([[1.04,.03,0],[0,.98,.02],[0,0,1.01]]).T).requires_grad_()
    E,r,H=m.evaluate(x,th,0,True)
    automatic,=torch.autograd.grad(E,x)
    assert torch.allclose(r,automatic,atol=2e-10,rtol=2e-8)
    rng=np.random.default_rng(29);direction=m.t(rng.normal(size=x.shape));direction/=direction.norm()
    eps=1e-6
    rp=m.evaluate(x.detach()+eps*direction,th,0)[1]
    rm=m.evaluate(x.detach()-eps*direction,th,0)[1]
    assert torch.allclose(H@direction.flatten(),((rp-rm)/(2*eps)).flatten(),atol=2e-7,rtol=2e-5)
    assert torch.allclose(H,H.T,atol=1e-9,rtol=1e-8)


def test_parameter_force_derivative():
    m=small_model();th=m.t(parameters(m)[0]).requires_grad_();x=m.rest.clone();x[:,0]*=1.03
    r=m.evaluate(x,th,0)[1];direction=m.t(np.random.default_rng(31).normal(size=x.shape))
    analytic,=torch.autograd.grad((r*direction).sum(),th)
    for index in [0,1,m.i_material,m.extra,m.extra+1,m.extra+2]:
        step=torch.zeros_like(th);step[index]=1e-5
        plus=m.evaluate(x,th.detach()+step,0)[1];minus=m.evaluate(x,th.detach()-step,0)[1]
        numerical=((plus-minus)*direction).sum()/(2e-5)
        assert torch.allclose(analytic[index],numerical,atol=1e-8,rtol=1e-4)
