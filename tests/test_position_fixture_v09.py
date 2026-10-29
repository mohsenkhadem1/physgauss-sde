import numpy as np
from scripts.position_ood.fixture import PositionFixture,controls

def fixture(**kw):return PositionFixture(2,20000,5000,8000,'crossed',device='cpu',**kw)

def test_position_target_energy_gradient_and_tangent():
    m=fixture(field_amplitude_J=.02);m.linear_solver='superlu';m.set_controls(controls('twist_x'))
    rng=np.random.default_rng(901);u=rng.normal(0,1e-5,m.rest.shape);d=rng.normal(size=u.shape);d/=np.linalg.norm(d);eps=1e-6
    state=m.evaluate(u,True);plus=m.evaluate(u+eps*d);minus=m.evaluate(u-eps*d)
    assert np.isclose((plus['potential_J']-minus['potential_J'])/(2*eps),np.sum(state['residual']*d),rtol=2e-5,atol=1e-9)
    np.testing.assert_allclose((plus['residual']-minus['residual']).ravel()/(2*eps),state['tangent']@d.ravel(),rtol=3e-5,atol=2e-7)

def test_rigid_translation_with_controller_positions_is_stress_free():
    m=fixture(gravity=(0,0,0));m.linear_solver='superlu';p=np.zeros((2,6));p[:,:3]=[.002,-.003,.001];m.set_controls(p)
    u=np.broadcast_to(p[0,:3],m.rest.shape).copy();state=m.evaluate(u)
    assert np.linalg.norm(state['residual'])<1e-9
    assert abs(state['elastic_J'])<1e-12

def test_gravity_and_reactions_balance_without_force_inputs():
    m=fixture(field_amplitude_J=.01);m.linear_solver='superlu';m.set_controls(controls('translate_y'))
    state=m.solve(force_tolerance_N=1e-9);sample=m.sampled(state)
    assert sample['residual_N']<1e-9 and sample['reaction_balance_N']<1e-8
    np.testing.assert_allclose(sample['body_force_N'],[0,0,-.96*9.81],atol=1e-12)
    assert sample['minimum_J']>.1
