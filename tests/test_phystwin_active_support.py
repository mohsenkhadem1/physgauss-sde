import numpy as np
from scripts.phystwin.active_support_reference import ActiveBlock
from scripts.phystwin.active_support_policy import Mixture


def test_half_face_area_and_load_conservation():
    full=ActiveBlock(2,3000,800,'left_full');half=ActiveBlock(2,3000,800,'right_lower')
    assert np.isclose(half.attachment_area,full.attachment_area/2)
    assert np.isclose(half.support_weights.sum(),1)
    assert np.all(half.rest[half.support_weights>0,2]<=1e-13)
    half.set_action('push_left')
    assert np.allclose(half.load.sum(0),[0,.03,0])


def test_partial_fixture_energy_tangent_and_force_balance():
    m=ActiveBlock(2,3000,800,'left_lower');m.set_action('held_oblique')
    rng=np.random.default_rng(2);u=rng.normal(size=m.rest.shape)*1e-5;d=rng.normal(size=u.shape);d/=np.linalg.norm(d)
    h=1e-6;s=m.evaluate(u,True);a=m.evaluate(u+h*d);b=m.evaluate(u-h*d)
    assert np.isclose((a['potential_J']-b['potential_J'])/(2*h),np.sum(s['residual']*d),rtol=1e-6,atol=1e-9)
    assert np.allclose((a['residual']-b['residual']).ravel()/(2*h),s['tangent']@d.ravel(),rtol=1e-5,atol=1e-6)
    solution=m.solve();u=solution['displacement_local_m']
    assert np.linalg.norm((m.K*m.support_weights[:,None]*u-m.load).sum(0))<1e-9


def test_expected_risk_for_single_gaussian_is_analytic():
    fields=np.zeros((4,2));prior=np.ones(4)/4;C=np.array([[4.,1.],[1.,2.]])
    m=Mixture(fields,prior,np.zeros(2),C,[0],[1],1.,'cpu')
    # Var(target | observed) = 2 - 1^2/(4+1), independently of observed value.
    assert np.allclose(m.risk(np.array([[-2.],[0.],[3.]])),1.8)
    result=m.infer(np.array([[3.]]))
    assert np.allclose(result['mean'],.6)
    assert np.allclose(result['sd']**2,1.8)


def test_probe_informativeness_and_utility_scoring_consistency():
    fields=np.array([[-.01,-.02],[-.01,-.02],[.01,.02],[.01,.02]])
    m=Mixture(fields,np.ones(4)/4,np.zeros(2),np.zeros((2,2)),[0],[1],.001,'cpu')
    y=np.array([[-.01],[.01],[0.]])
    result=m.infer(y)
    assert np.allclose(m.risk(y),np.mean(result['sd']**2,axis=1),atol=1e-14)
    assert m.risk(y)[0]<1e-12 and np.isclose(m.risk(y)[2],.0004)
