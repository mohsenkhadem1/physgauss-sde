import numpy as np
import pytest
from scripts.phystwin.sequential_fixture import SequentialFixture,LAYOUTS,ACTIONS

@pytest.mark.parametrize('layout',LAYOUTS)
def test_consistent_patches_and_forces(layout):
 m=SequentialFixture(2,4000,300,900,layout,device='cpu')
 assert np.isclose(m.fixture_scalar.sum(),1200)
 assert np.allclose(m.fixture_scalar.toarray(),m.fixture_scalar.toarray().T)
 assert np.isclose(m.volume_weights.sum(),1)
 for action in ACTIONS:
  m.set_action(action);assert np.isfinite(m.load).all()
 m.set_action('stretch');assert np.allclose(m.load.sum(0),[0,0,-.02])
 m.set_action('twist');assert np.allclose(m.load.sum(0),[0,0,-.02])

def test_new_boundary_energy_gradient_and_tangent():
 m=SequentialFixture(2,4000,300,900,'crossed',device='cpu')
 m.set_action('twist');rng=np.random.default_rng(17);u=rng.normal(size=m.rest.shape)*1e-5;d=rng.normal(size=u.shape);d/=np.linalg.norm(d);h=1e-6
 state=m.evaluate(u,True);plus=m.evaluate(u+h*d);minus=m.evaluate(u-h*d)
 assert np.isclose((plus['potential_J']-minus['potential_J'])/(2*h),np.sum(state['residual']*d),rtol=1e-5,atol=1e-9)
 assert np.allclose((plus['residual']-minus['residual']).ravel()/(2*h),state['tangent']@d.ravel(),rtol=1e-5,atol=1e-7)
