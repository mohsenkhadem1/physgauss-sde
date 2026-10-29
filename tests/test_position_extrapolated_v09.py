import numpy as np
from scripts.position_ood.extrapolated import ExtrapolatedPositionFEM
from scripts.position_ood.fixture import controls

def test_extrapolated_field_and_adjoint_are_consistent():
    m=ExtrapolatedPositionFEM(1,2,'cpu');np.testing.assert_allclose(m.interpolation@m.coarse.geo.rest,m.geo.rest,atol=1e-14)
    theta=m.tensor([*np.log([20000,5000,8000]),0,0,-9.81,.1,.3]).requires_grad_();pose=m.tensor(controls('translate_y'));value=m.predict(theta,pose,1).square().mean();value.backward()
    a=theta.detach().clone();b=a.clone();eps=1e-4;a[6]+=eps;b[6]-=eps;fd=(m.predict(a,pose,1).square().mean()-m.predict(b,pose,1).square().mean())/(2*eps)
    np.testing.assert_allclose(theta.grad[6],fd,rtol=3e-3,atol=1e-8)
