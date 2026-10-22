from itertools import product
from math import factorial
import numpy as np
import torch
from experiments.campaign_p2_keast_quadrature import positive_keast24,KeastBulkP2Geometry
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_streaming import StreamingTwoLevelP2


def test_all_barycentric_monomials_through_degree6_and_positive_symmetry():
    b,w=positive_keast24()
    assert len(b)==24 and (b>0).all() and (w>0).all()
    assert abs(w.sum()-1)<2e-15
    eps=np.finfo(float).eps
    np.testing.assert_allclose(b.sum(1),1,atol=3*eps/(1-3*eps),rtol=0)
    for alpha in product(range(7),repeat=4):
        degree=sum(alpha)
        if degree>6:continue
        exact=6*np.prod([factorial(k) for k in alpha])/factorial(degree+3)
        value=w@np.prod(b**np.asarray(alpha),axis=1)
        assert abs(value-exact)<3e-15


def test_exact80_energy_force_fullHVP_mass_blocks_load_and_guard_parity():
    torch.set_num_threads(2)
    m=KeastBulkP2Geometry((1,1,1),young=4500,gravity=9.81,pattern='axial_pull')
    old=ExactBulkP2Geometry((1,1,1),young=4500,gravity=9.81,pattern='axial_pull')
    np.testing.assert_array_equal(m.check_bary[:10],old.check_bary)
    np.testing.assert_array_equal(m.check_bary[10:],old.bary)
    np.testing.assert_array_equal(m.load,old.load)
    a=StreamingTwoLevelP2(m);b=StreamingTwoLevelP2(old)
    X=m.rest
    u=torch.tensor(np.column_stack([.1*X[:,0]+.3*X[:,1]*X[:,2],
                                   -.04*X[:,1]+.3*X[:,0]*X[:,2],
                                   .06*X[:,2]+.4*X[:,0]*X[:,1]]),dtype=torch.float64)
    v=torch.tensor(np.random.default_rng(6024).normal(size=X.shape),dtype=torch.float64)
    ca,cb=a.evaluate(u),b.evaluate(u)
    torch.testing.assert_close(ca['energy'],cb['energy'],atol=2e-14,rtol=2e-12)
    torch.testing.assert_close(ca['gradient'],cb['gradient'],atol=2e-11,rtol=2e-12)
    torch.testing.assert_close(a.tangent(ca,v),b.tangent(cb,v),atol=2e-9,rtol=2e-12)
    torch.testing.assert_close(a.nodal_blocks(ca,17),b.nodal_blocks(cb,17),atol=2e-10,rtol=2e-12)
    torch.testing.assert_close(a.mass(v),b.mass(v),atol=0,rtol=0)
