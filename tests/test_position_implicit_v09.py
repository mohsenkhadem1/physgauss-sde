import numpy as np
import torch
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.position_ood.fixture import PositionFixture,controls

def test_dense_backend_matches_independent_sparse_q2():
    dense=DensePositionFEM(2,'cpu');sparse=PositionFixture(2,20000,5000,8000,'crossed',device='cpu',field_amplitude_J=.02);sparse.linear_solver='superlu'
    pose=controls('translate_y');sparse.set_controls(pose);u=np.random.default_rng(91).normal(0,1e-5,sparse.rest.shape)
    theta=dense.tensor([*np.log([20000,5000,8000]),0,0,-9.81,.02]);e,r,H=dense.evaluate(dense.tensor(u),theta,dense.tensor(pose),1,True);ref=sparse.evaluate(u,True)
    np.testing.assert_allclose(e.detach(),ref['potential_J'],atol=1e-11)
    np.testing.assert_allclose(r.detach(),ref['residual'],atol=1e-10)
    np.testing.assert_allclose(H.detach(),ref['tangent'].toarray(),atol=1e-9)

def test_implicit_gradient_matches_equilibrium_finite_difference():
    m=DensePositionFEM(1,'cpu');theta=m.tensor([*np.log([20000,5000,8000]),0,0,-9.81,.02]).requires_grad_();pose=m.tensor(controls('translate_y'))
    w=m.tensor(np.random.default_rng(901).normal(size=m.rest.shape));loss=(m.predict(theta,pose,1)*w).sum();loss.backward()
    for k in [0,2,4,6]:
        eps=1e-4;a=theta.detach().clone();b=a.clone();a[k]+=eps;b[k]-=eps
        with torch.no_grad():fd=((m.predict(a,pose,1)-m.predict(b,pose,1))*w).sum()/(2*eps)
        np.testing.assert_allclose(theta.grad[k],fd,rtol=3e-3,atol=2e-6)
