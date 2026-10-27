import numpy as np
from scripts.phystwin.controlled_reference import ControlledBlock


def test_robin_reference_rest_and_known_mass():
    m=ControlledBlock(2,2000.,1000.)
    s=m.evaluate(np.zeros_like(m.rest),True)
    assert np.linalg.norm(s['residual'])<1e-12
    assert np.isclose(m.node_mass.sum(),.96)
    assert np.allclose(s['J'],1.)


def test_energy_gradient_and_tangent_directions():
    m=ControlledBlock(2,2000.,1000.)
    rng=np.random.default_rng(4)
    u=rng.normal(size=m.rest.shape)*1e-4
    d=rng.normal(size=u.shape);d/=np.linalg.norm(d)
    m.target=np.array([.002,.001,0])
    s=m.evaluate(u,True);h=1e-6
    plus=m.evaluate(u+h*d);minus=m.evaluate(u-h*d)
    energy_fd=(plus['potential_J']-minus['potential_J'])/(2*h)
    assert np.isclose(energy_fd,np.sum(s['residual']*d),rtol=1e-6,atol=1e-9)
    force_fd=(plus['residual']-minus['residual']).ravel()/(2*h)
    assert np.allclose(force_fd,s['tangent']@d.ravel(),rtol=1e-5,atol=1e-6)


def test_solved_end_attachment_force_balance():
    m=ControlledBlock(2,8000.,64.);m.target=np.array([0,.009,0])
    answer=m.solve();u=answer['displacement_local_m']
    reaction=(m.support*m.left[:,None]*u+m.actuator*m.right[:,None]*(u-m.target)).sum(0)
    assert np.linalg.norm(answer['residual'])<1e-9
    assert np.linalg.norm(reaction)<1e-9
    assert answer['J'].min()>.9
