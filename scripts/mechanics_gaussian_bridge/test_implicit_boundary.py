import numpy as np,torch
from scripts.real_transfer_v19.neural_fem import NeuralFEM,parameters
from scripts.pidg_mechanics.fit_mechanics import vertices
from scripts.mechanics_gaussian_bridge.implicit_boundary import BoundaryEquilibrium
v=vertices()*.1;weights=np.ones((1,len(v)));data=dict(nodes=v,controller_targets=v[None,None]+[.01,0,0],attachment_weights=weights,attachment_active=np.ones((1,1)),floor=-100.)
m=NeuralFEM(data,device='cpu',materials=1);th,_=parameters(m);th[0]=-20;th[m.i_floor]=-20;theta=m.t(th).requires_grad_();target=m.targets.clone().requires_grad_();x=BoundaryEquilibrium.apply(theta,target,m,0,None);loss=x[:,0].mean();gt,gc=torch.autograd.grad(loss,(theta,target));eps=1e-5
values=[]
for sign in [-1,1]:
 moved=target.detach().clone();moved[...,0]+=sign*eps;values.append(float(BoundaryEquilibrium.apply(theta.detach(),moved,m,0,None)[:,0].mean()))
fd=(values[1]-values[0])/(2*eps);ad=float(gc[...,0].sum());np.testing.assert_allclose(fd,ad,rtol=1e-4,atol=1e-6);assert torch.isfinite(gt).all();print('PASS target adjoint',ad,'finite difference',fd)
