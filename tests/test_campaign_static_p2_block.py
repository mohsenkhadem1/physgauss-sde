import numpy as np
import pytest
from experiments.campaign_static_p2_block import QuadraticBlock,shapes,shape_gradients,duffy
from experiments.campaign_static_uniaxial_bar import analytic_solution
from experiments.campaign_static_supported_block import SIZE


def test_partition_affine_compatible_quadrature_and_signed_body_load():
    m=QuadraticBlock((1,1,1),traction=1.)
    for b in m.bary:
        N=shapes(b);g=shape_gradients(b[None],m.linear_gradients)
        np.testing.assert_allclose(N.sum(),1.,atol=2e-15)
        np.testing.assert_allclose(g.sum(1),0.,atol=2e-13)
        np.testing.assert_allclose(np.einsum('eai,eaJ->eiJ',m.rest[m.elements],g),np.broadcast_to(np.eye(3),(6,3,3)),atol=2e-14)
    assert m.weights.sum()==pytest.approx(1.,abs=1e-15)
    assert m.body_integral_weights.min()<0
    np.testing.assert_allclose(m.load.sum(0),[1.,0.,-9.4176],atol=1e-14)
    expected=np.cross([.24,0,0],[1,0,0])+np.cross([.12,0,0],[0,0,-9.4176])
    np.testing.assert_allclose(np.cross(m.rest,m.load).sum(0),expected,atol=1e-14)


def test_finite_rotation_energy_gradient_hessian_and_invalid_J():
    m=QuadraticBlock((1,1,1),gravity=0.,young=5000.)
    angle=.8;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    U,g,_,J=m.elastic(m.rest@R.T+[.2,-.1,.3],True)
    assert abs(U)<1e-12 and np.linalg.norm(g)<1e-10
    rng=np.random.default_rng(2);u=rng.normal(size=m.rest.shape)*.0001;v=rng.normal(size=u.shape);v/=np.linalg.norm(v);eps=1e-7
    U,g,H,J=m.elastic(m.rest+u,True)
    plus=m.elastic(m.rest+u+eps*v);minus=m.elastic(m.rest+u-eps*v)
    assert (plus[0]-minus[0])/(2*eps)==pytest.approx(np.sum(g*v),rel=1e-7,abs=2e-9)
    np.testing.assert_allclose((plus[1]-minus[1]).ravel()/(2*eps),H@v.ravel(),rtol=2e-7,atol=2e-7)
    np.testing.assert_allclose(H.toarray(),H.T.toarray(),atol=2e-12)
    with pytest.raises(ValueError,match='Jacobian'):m.elastic(m.rest@np.diag([-.2,1,1]))


def test_homogeneous_uniaxial_oracle_and_order3_order4_parity():
    states=[]
    for order in [3,4]:
        m=QuadraticBlock((1,2,2),young=5000.,gravity=0.,traction=1.,order=order)
        m.mask[:]=False;m.mask[:,0]=abs(m.rest[:,0])<1e-13;m.mask[:,1]=abs(m.rest[:,1])<1e-13;m.mask[:,2]=abs(m.rest[:,2])<1e-13;m.free=np.flatnonzero(~m.mask.ravel())
        a=m.solve();o=analytic_solution(1.,size=SIZE)
        exact=m.rest*np.array([o['stretch'],o['lateral_stretch'],o['lateral_stretch']])
        np.testing.assert_allclose(a['positions'],exact,atol=2e-11,rtol=0)
        points=np.array([[.05,.01,.02],[.12,-.02,.01],[.24,0,0]])
        np.testing.assert_allclose(m.query(a['positions'],points),points*np.array([o['stretch'],o['lateral_stretch'],o['lateral_stretch']]),atol=2e-11)
        states.append(a['positions'])
    np.testing.assert_allclose(states[0],states[1],atol=2e-12)
