import numpy as np
from scripts.position_ood.study import interpolate_samples
from scripts.phystwin.tensor_fixture import TensorFixture

def test_refinement_preserves_quadratic_field_on_graded_mesh():
    # n=6 is deliberately not uniform in z; n=1,2,4 hide this error.
    coarse=TensorFixture(6,1.,1.,'left_full',device='cpu',x_power=1.)
    query=np.random.default_rng(29).uniform([0,-.04,-.04],[.24,.04,.04],(100,3))
    def polynomial(x):return np.c_[x[:,0]**2+x[:,2],x[:,1]*x[:,2],x[:,2]**2-2*x[:,0]]
    np.testing.assert_allclose(interpolate_samples(polynomial(coarse.rest),query,n=6),polynomial(query),atol=1e-14)

def test_reused_tangent_reaches_same_nonlinear_equilibrium():
    from scripts.position_ood.material_field import MaterialPositionFixture
    from scripts.position_ood.fixture import controls
    from scripts.position_ood.reference_solver import solve
    m=MaterialPositionFixture(2,20000,5000,8000,'crossed',device='cpu',material_gradient=.5,field_amplitude_J=.1);m.linear_solver='superlu'
    m.set_controls(controls('translate_y'));first=m.solve(force_tolerance_N=1e-9)
    m.set_controls(controls('twist_x'));a=solve(m,initial=first['displacement_local_m']);b=m.solve(initial=first['displacement_local_m'],force_tolerance_N=1e-9)
    assert np.linalg.norm(a['residual'])<1e-9
    np.testing.assert_allclose(a['displacement_local_m'],b['displacement_local_m'],atol=1e-10)
    assert a['tangent_assemblies']<a['iterations']
