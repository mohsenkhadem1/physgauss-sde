import numpy as np,torch
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.pidg_mechanics.fit_mechanics import vertices
from scripts.mechanics_gaussian_bridge.tetra import TetraBinding
v=vertices()*.1+np.array([1.,2.,3.]);data=dict(nodes=v,controller_targets=v[None,None],attachment_weights=np.ones((1,len(v))),attachment_active=np.ones((1,1)),floor=-100.)
m=NeuralFEM(data,device='cpu',materials=1);centers=v[1:]*.99+v[:1]*.01;cov=np.tile(np.diag([1e-4,2e-4,3e-4]),(12,1,1));binding=TetraBinding(m,centers,cov);F=np.array([[1.08,.08,0],[0,.95,.02],[0,0,1.03]]);t=np.array([.1,-.2,.05]);world=v@F.T+t;x=m.t((world-m.origin)/m.scale).requires_grad_();state=binding.transport(x)
np.testing.assert_allclose(state['centers'].detach(),centers@F.T+t,atol=1e-12);np.testing.assert_allclose(state['covariance'].detach(),F@cov@F.T,atol=1e-12);assert torch.autograd.gradcheck(lambda q:binding.transport(q)['centers'],(x,),fast_mode=True)
print('PASS tetra affine position/covariance exactness and node derivatives')
