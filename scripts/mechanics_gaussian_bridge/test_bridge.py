import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,Q2Mechanics
from scripts.compositional_dynamics_v17.gaussian_transport import bind_gaussians,transport_gaussians
from scripts.mechanics_gaussian_bridge.bridge import Q2Binding

def test_material_transport():
 m=Q2Mechanics(n=1);rng=np.random.default_rng(42);centers=m.geometry.map(rng.uniform(.05,.95,(12,3)));cov=np.tile(np.diag([1e-6,2e-6,3e-6]),(12,1,1));b=Q2Binding(m,centers,cov)
 F=np.array([[1.1,.07,0],[0,.94,.02],[0,0,1.03]]);x=m.rest@F.T+[.01,-.02,.04]
 state=b.transport(torch.tensor(x,dtype=torch.double,requires_grad=True));np.testing.assert_allclose(state['centers'].detach(),centers@F.T+[.01,-.02,.04],atol=1e-12)
 old=transport_gaussians(m,x-m.rest,bind_gaussians(m.geometry,centers,cov))
 np.testing.assert_allclose(state['covariance'].detach(),old['covariance_m2'],atol=1e-12)
 np.testing.assert_allclose(state['rotation'].detach(),old['material_rotation'],atol=1e-12)
 identity=torch.tensor(m.rest,dtype=torch.double,requires_grad=True)
 assert torch.autograd.gradcheck(lambda nodes:tuple(b.transport(nodes)[k] for k in ['centers','covariance','rotation']),(identity,),fast_mode=True,atol=1e-5)
 bad=torch.tensor(m.rest*np.array([-1,1,1]),dtype=torch.double)
 try:b.transport(bad)
 except ValueError:pass
 else:raise AssertionError('Inversion must fail')
if __name__=='__main__':test_material_transport();print('PASS affine exactness, existing-transport parity, identity gradients, inversion rejection')
