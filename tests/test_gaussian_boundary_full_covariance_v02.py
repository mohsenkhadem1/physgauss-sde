import math

import pytest
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

from scripts.gaussian_boundary_flow_v02.full_covariance import FullCovarianceMixtureRegressor
from scripts.gaussian_boundary_flow_v02.probabilistic import GaussianMixtureRegressor


def test_density_matches_torch_full_covariance_mixture():
    torch.manual_seed(11)
    m=FullCovarianceMixtureRegressor(4,3,width=12).double()
    c=torch.randn(7,4,dtype=torch.float64);y=torch.randn(7,3,dtype=torch.float64)
    logits,mean,L=m(c)
    reference=MixtureSameFamily(Categorical(logits=logits),MultivariateNormal(mean,scale_tril=L))
    assert torch.allclose(m.log_prob(y,c),reference.log_prob(y),atol=1e-12,rtol=1e-12)
    assert torch.all(L.diagonal(dim1=-2,dim2=-1)>0)
    assert torch.equal(L,torch.tril(L))


def test_samples_reproduce_known_correlated_moments():
    m=FullCovarianceMixtureRegressor(2,2,components=1,width=4).double()
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        # logits; means; raw triangle [L00,L10,L11].
        m.network[-1].bias.copy_(torch.tensor([0.,1.,-2.,math.log(2.),10*math.atanh(.2),math.log(.5)],dtype=torch.float64))
    c=torch.zeros(1,2,dtype=torch.float64)
    a=m.sample(c,40000,torch.Generator().manual_seed(921));b=m.sample(c,40000,torch.Generator().manual_seed(921))
    assert torch.equal(a,b)
    assert a.shape==(1,40000,2)
    # L=[[2,0],[1,.5]], covariance[[4,2],[2,1.25]].
    assert torch.allclose(a.mean(1),torch.tensor([[1.,-2.]],dtype=torch.float64),atol=.03)
    assert torch.allclose(torch.cov(a[0].T),torch.tensor([[4.,2.],[2.,1.25]],dtype=torch.float64),atol=.08)


def test_likelihood_gradients_reach_off_diagonals_and_match_finite_difference():
    torch.manual_seed(992)
    m=FullCovarianceMixtureRegressor(3,2,width=7).double()
    c=torch.randn(3,3,dtype=torch.float64,requires_grad=True);y=torch.randn(3,2,dtype=torch.float64)
    loss=m.loss(y,c);loss.backward()
    head=m.network[-1].bias.grad.reshape(2,6)
    assert head[:,4].abs().sum()>0  # raw off-diagonal triangle entry
    shift=torch.zeros_like(c);shift[1,2]=1e-5
    finite=(m.loss(y,c.detach()+shift)-m.loss(y,c.detach()-shift))/2e-5
    assert torch.allclose(c.grad[1,2],finite,atol=1e-9,rtol=1e-5)


def test_extreme_raw_scales_remain_finite():
    m=FullCovarianceMixtureRegressor(2,6,width=4)
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.network[-1].bias.fill_(-1000.)
    assert torch.isfinite(m.log_prob(torch.ones(1,6),torch.zeros(1,2))).all()


def test_capacity_and_invalid_contracts():
    m=FullCovarianceMixtureRegressor(144);d=GaussianMixtureRegressor(144)
    count=lambda x:sum(p.numel() for p in x.parameters())
    assert count(m)==141516
    assert abs(count(m)/count(d)-1)<.002
    with pytest.raises(ValueError):m(torch.zeros(2,143))
    with pytest.raises(ValueError):m.sample(torch.zeros(2,144),count=0)
    with pytest.raises(ValueError):FullCovarianceMixtureRegressor(2,off_diagonal_limit=0)
