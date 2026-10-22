import numpy as np
import pytest
import torch
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_graded_coarse import StreamingGradedCoarseP2


@pytest.fixture(scope='module')
def system():
    torch.set_num_threads(2);m=ExactBulkP2Geometry((1,1,1));t=StreamingGradedCoarseP2(m)
    return m,t


def test_graded_coarse_HPD_geometry_adjoint_and_SPD(system):
    m,t=system
    assert t.coarse_free.numel()==3888 and np.isclose(t.coarse_first_x_m,.00375)
    assert not hasattr(t,'grad') and len(t.bary)==80
    torch.linalg.cholesky(t.Kc);torch.linalg.cholesky(t.Mc)
    torch.testing.assert_close(torch.sparse.mm(t.P,torch.ones((len(t.coarse_rest),1),dtype=t.dtype)),torch.ones((len(m.rest),1),dtype=t.dtype),atol=2e-13,rtol=0)
    torch.testing.assert_close(torch.sparse.mm(t.P,t.coarse_rest),t.rest,atol=2e-14,rtol=0)
    rng=np.random.default_rng(846);c=torch.tensor(rng.normal(size=t.coarse_free.numel()),dtype=t.dtype);r=torch.tensor(rng.normal(size=m.rest.shape),dtype=t.dtype)
    assert abs(float((r*t.prolong(c)).sum()-(t.restrict(r)*c).sum()))<1e-11
    assert not bool((t.prolong(c)[t.mask]!=0).any())
    cache=t.evaluate(torch.zeros_like(t.rest));B=t.make_preconditioner(cache,31.)
    z=torch.tensor(rng.normal(size=m.rest.shape),dtype=t.dtype);Br,Bz=B(r),B(z)
    assert float((r*Br).sum())>0
    assert abs(float((r*Bz).sum()-(z*Br).sum()))<1e-9


def test_original_fine_operator_linear_and_static_parity(system):
    m,t=system;old=StreamingTwoLevelP2(m);rng=np.random.default_rng(837)
    u=torch.tensor(rng.normal(0,1e-4,m.rest.shape),dtype=t.dtype);u[t.mask]=0
    v=torch.tensor(rng.normal(size=m.rest.shape),dtype=t.dtype);v[t.mask]=0
    a,b=t.evaluate(u),old.evaluate(u)
    torch.testing.assert_close(a['gradient'],b['gradient'],atol=0,rtol=0)
    torch.testing.assert_close(t.tangent(a,v),old.tangent(b,v),atol=0,rtol=0)
    x,k=t.cg(a,v,1000.,rtol=1e-12,atol=1e-12)
    R=(old.tangent(b,x)+1000*old.mass(x)-v)[~t.mask]
    assert float(torch.linalg.vector_norm(R))<1e-9
    new=t.solve();baseline=old.solve()
    assert float(torch.linalg.vector_norm(new['residual']))<=1e-10
    torch.testing.assert_close(new['positions'],baseline['positions'],atol=3e-11,rtol=0)
    assert all(r['preconditioner']=='additive_exactnodeblock_plus_graded8x4x4_restP2' for r in t.linear_records)


def test_scope_not_broadened():
    m=P2Geometry((1,1,1),support='right_face')
    with pytest.raises(ValueError,match='full-left'):StreamingGradedCoarseP2(m)
