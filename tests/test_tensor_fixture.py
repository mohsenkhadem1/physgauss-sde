import numpy as np
import pytest
from scripts.phystwin.tensor_fixture import basis,TensorFixture


def test_tensor_basis_and_mixed_polynomials():
    nodes=np.stack(np.meshgrid(*[[-1.,0.,1.]]*3,indexing='ij'),-1).reshape(-1,3)
    assert np.allclose(basis(nodes)[0],np.eye(27))
    p=np.random.default_rng(4).uniform(-1,1,(40,3));N,G=basis(p)
    assert np.allclose(N.sum(1),1) and np.allclose(G.sum(1),0)
    values=nodes[:,0]**2*nodes[:,2]**2
    assert np.allclose(N@values,p[:,0]**2*p[:,2]**2)
    assert np.allclose(np.einsum('pai,a->pi',G,values),np.c_[2*p[:,0]*p[:,2]**2,np.zeros(len(p)),2*p[:,2]*p[:,0]**2])


def test_tensor_surface_material_and_newton():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    for layout in ['left_full','right_lower']:
        m=TensorFixture(2,2000.,600.,layout)
        assert np.isclose(m.volumes.sum(),.24*.08*.08)
        assert np.isclose(m.attachment_area,.08*(.04 if layout.endswith('lower') else .08))
        assert np.isclose(m.left.sum(),1) and np.isclose(m.right.sum(),1)
        t=np.tile([.001,.002,-.003],(len(m.rest),1))
        assert np.allclose((m.fixture_scalar@t).sum(0),600*t[0])
        # Independent integral of z^2 over the supported rectangle.
        assert np.isclose(m.rest[:,2]@(m.fixture_scalar@m.rest[:,2]),600*.04**2/3)
        F=np.diag([1.02,.99,1.01]);u=m.rest@(F-np.eye(3)).T
        U,g,H,J=m.elastic(m.rest+u,True)
        mu=2000/(2*1.35);lam=2000*.35/(1.35*.3);lj=np.log(np.linalg.det(F))
        energy=(mu/2*((F*F).sum()-3)-mu*lj+lam/2*lj**2)*m.volumes.sum()
        assert np.isclose(U,energy,rtol=1e-10)
        direction=np.random.default_rng(8).normal(size=u.shape)*.01;eps=1e-5
        up,gp,_,_=m.elastic(m.rest+u+eps*direction,False)
        um,gm,_,_=m.elastic(m.rest+u-eps*direction,False)
        assert np.isclose((up-um)/(2*eps),np.sum(g*direction),rtol=1e-6,atol=1e-10)
        assert np.allclose((gp-gm).ravel()/(2*eps),H@direction.ravel(),rtol=1e-5,atol=1e-6)
        p=np.array([[.011,.002,-.007],[.22,-.03,.01]])
        assert np.allclose(m.query_field(m.rest**2,p),p**2)
        m.set_action('push_left' if layout.startswith('left') else 'push_right');state=m.solve()
        assert np.linalg.norm(state['residual'])<1e-10
        if layout.endswith('full'):
            assert np.allclose(state['displacement_local_m'],[0,.03/600,0],atol=1e-12)
        m.clear_linear_cache()
