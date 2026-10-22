import numpy as np
from scipy.integrate import quad
from experiments.campaign_c38_bar_posterior_z import conditional_nodes,BarPosteriorBatch,C_LO,C_HI


def test_standardized_nodes_resolve_boundary_far_tail_partial_mass():
    for m in (-.005,.001,.0025,.005,.01,.012,.02):
        for upper in (C_HI,.004,C_LO+1e-7):
            c,w,mass=conditional_nodes(m,.0002,upper,64);assert (c>C_LO-1e-16).all() and (c<min(upper,C_HI)+1e-16).all()
            mode=np.clip(m,C_LO,upper);shift=.5*((mode-m)/.0002)**2
            f=lambda x:np.exp(shift-.5*((x-m)/.0002)**2)
            Z=quad(lambda x:f(x)/x,C_LO,upper,epsabs=1e-11,epsrel=1e-11)[0]
            estimate=np.exp(shift+mass)*np.sqrt(2*np.pi)*.0002*np.sum(w/c)
            assert abs(estimate/Z-1)<1e-9
            # Centered conditional variance must be accurate, not only mass.
            weights=w/c;weights/=weights.sum();mean=weights@c;var=weights@((c-mean)**2)
            expected=quad(lambda x:f(x)/x*(x-mean)**2,C_LO,upper,epsabs=1e-24,epsrel=1e-10)[0]/Z
            assert abs(var-expected)<max(1e-22,expected*1e-8)


def test_two_conditional_orders_and_independent_2d_control():
    from experiments.run_campaign_c38_bar_active import summarize,agreement,independent
    low=BarPosteriorBatch([[.005]],[1.],[0],rtol=1e-6,conditional_order=32)
    high=BarPosteriorBatch([[.005]],[1.],[0],rtol=1e-8,conditional_order=64)
    sl=summarize(low);sh=summarize(high);assert agreement(sl,sh,low,high)['passed']
    assert independent(high,0,sh)['passed']


def test_resolved_2d_boundary_cases_and_old_high_order_field_moments():
    from experiments.run_campaign_c38_bar_active import summarize,independent
    from experiments.campaign_c38_bar_posterior import BarPosteriorBatch as ProbabilityPosterior
    from experiments.campaign_c38_bar_reference import material_quadrature
    for measured in (.00245,.0101):
        p=BarPosteriorBatch([[measured]],[1.],[0],rtol=1e-8,conditional_order=64)
        assert independent(p,0,summarize(p))['passed']
    p=BarPosteriorBatch([[.005]],[1.],[0],rtol=1e-8,conditional_order=64)
    old=ProbabilityPosterior([[.005]],[1.],[0],rtol=1e-8,conditional_order=128)
    a,b=p.field_moments(material_quadrature());x,y=old.field_moments(material_quadrature())
    assert np.max(abs(a-x))<2e-8 and np.max(abs(b-y))<2e-8
