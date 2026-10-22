import numpy as np
import torch
from experiments.campaign_p2_torch import TorchP2,consistent_t10
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_static_p2_block import duffy,shapes


def test_exact_consistent_mass_matches_independent_degree_four_quadrature():
    b,w=duffy(4);N=shapes(b);numerical=np.einsum('q,qa,qb->ab',w,N,N)
    np.testing.assert_allclose(consistent_t10(),numerical,atol=2e-15,rtol=0)
    assert np.linalg.eigvalsh(consistent_t10()).min()>0
    np.testing.assert_allclose(consistent_t10().sum(1),[-.05]*4+[.2]*6,atol=1e-16)
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1);t=TorchP2(m)
    a=torch.ones_like(t.rest);assert abs(float((a*t.mass(a)).sum())/3-.96)<1e-14


def test_energy_gradient_full_hvp_and_diagonal_match_independent_CPU():
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1);t=TorchP2(m);rng=np.random.default_rng(113)
    u=rng.normal(size=m.rest.shape)*2e-5;v=rng.normal(size=m.rest.shape);E,g,H,J=m.elastic(m.rest+u,True)
    a=t.evaluate(torch.tensor(u));np.testing.assert_allclose(float(a['energy']),E,atol=2e-17,rtol=2e-9)
    np.testing.assert_allclose(a['gradient'].numpy(),g,atol=2e-13,rtol=2e-9)
    np.testing.assert_allclose(t.tangent(a,torch.tensor(v)).numpy().ravel(),H@v.ravel(),atol=2e-11,rtol=2e-12)
    np.testing.assert_allclose(t.diagonal(a).numpy().ravel(),H.diagonal(),atol=2e-12,rtol=2e-13)
    eps=1e-7;plus=t.evaluate(torch.tensor(u+eps*v));minus=t.evaluate(torch.tensor(u-eps*v))
    np.testing.assert_allclose((plus['gradient']-minus['gradient']).numpy().ravel()/(2*eps),H@v.ravel(),rtol=2e-8,atol=1e-6)


def test_static_and_inertial_original_equilibria():
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1);t=TorchP2(m)
    a=m.solve();b=t.solve();np.testing.assert_allclose(b['positions'].numpy(),a['positions'],atol=2e-11,rtol=0)
    h=.01;target=torch.zeros_like(t.rest);c=t.solve(target=target,effective_dt=h,force_tolerance=1e-10)
    r=c['cache']['gradient']+t.mass(c['displacement'])/h**2-t.force
    assert float(torch.linalg.vector_norm(r[~t.mask]))<1e-10
    np.testing.assert_array_equal(c['displacement'].numpy()[m.mask],0)
