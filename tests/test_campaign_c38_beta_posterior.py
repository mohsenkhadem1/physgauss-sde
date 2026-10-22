import numpy as np
import torch
from experiments.campaign_c38_beta_posterior import *


def test_observation_order_mask_and_label_free_context():
    f=np.array([[1.,2.],[1.,np.nan]]);s=np.array([[1.,0.],[0.,np.nan]]);y=np.array([[.01,.02],[.005,np.nan]]);mask=np.array([[1,1],[1,0]],dtype=bool)
    first=canonical_inputs(f,s,y,mask);second=canonical_inputs(f[:,::-1],s[:,::-1],y[:,::-1],mask[:,::-1]);np.testing.assert_array_equal(first,second)
    assert np.isfinite(first).all() and first.shape==(2,10)
    assert not {'theta','material','role'}&set(__import__('inspect').signature(canonical_inputs).parameters)


def test_uniform_prior_density_marginals_and_finite_gradient():
    params=dict(logits=torch.zeros(3,16,dtype=torch.float64),alpha=torch.ones(3,16,2,dtype=torch.float64),beta=torch.ones(3,16,2,dtype=torch.float64));x=torch.tensor([[.1,.9],[.5,.5],[.99,.01]],dtype=torch.float64)
    torch.testing.assert_close(joint_log_prob(x,params),torch.zeros(3,dtype=torch.float64),atol=1e-14,rtol=0)
    np.testing.assert_allclose(marginal_cdf(params,x.numpy()),x,atol=1e-14)
    np.testing.assert_allclose(marginal_quantiles(params),np.broadcast_to([.05,.5,.95],(3,2,3)),atol=1e-12)
    model=BetaPosterior();p=model(torch.zeros(3,10));loss=-joint_log_prob(x,p).mean();loss.backward();assert all(a.grad is None or torch.isfinite(a.grad).all() for a in model.parameters())


def test_normalized_beta_quadrature_and_continuous_variance():
    for a,b in ((1.,1.),(.5,.5),(.001,2.),(2.,.001),(1200.,1500.),(.001,160000.),(160000.,.001),(15000.,12.)):
        x,w=beta_quadrature(a,b,16);np.testing.assert_allclose(w@x,a/(a+b),atol=2e-13)
        np.testing.assert_allclose(w@((x-w@x)**2),a*b/((a+b)**2*(a+b+1)),atol=2e-14)
    p=dict(logits=np.zeros((1,16)),alpha=np.ones((1,16,2)),beta=np.ones((1,16,2)))
    x,w=mixture_quadrature(p,0,order=12);np.testing.assert_allclose(w.sum(),1,atol=1e-14);assert len(x)==16*12**2
    np.testing.assert_allclose(w@x,[.5,.5],atol=1e-14)
    mean,std=field_moments(p,order=8);assert np.isfinite(mean).all() and (std>0).all()


def test_skewed_high_concentration_quantiles_match_independent_scipy():
    from scipy.stats import beta
    a=np.array([.001,1200.,15000.]);b=np.array([2.,1500.,12.]);p=dict(logits=np.zeros((3,16)),alpha=np.broadcast_to(a[:,None,None],(3,16,2)),beta=np.broadcast_to(b[:,None,None],(3,16,2)))
    q=marginal_quantiles(p)
    for i in range(3):
        expected=beta.ppf([.05,.5,.95],a[i],b[i]);np.testing.assert_allclose(q[i,0],expected,atol=1e-12,rtol=0)
