import numpy as np
from scipy.integrate import quad
from scripts.compositional_neural_mechanics.contact_resolution import ResolvedContactQ2,exact_flat_sphere_patch
from scripts.compositional_neural_mechanics.contact_mechanics import ContactQ2Mechanics,RigidSphere


def test_resolution16_preserves_original_operator():
    c=RigidSphere(center=(.12,0,-.07),radius=.045)
    a=ContactQ2Mechanics(n=1,surface_order=16,colliders=(c,));b=ResolvedContactQ2(n=1,surface_order=16,colliders=(c,));u=np.zeros_like(a.rest)
    for i in (0,1):np.testing.assert_array_equal(a.contact_response(a.engine.tensor(u))[i],b.contact_response(b.engine.tensor(u))[i])


def test_exact_spherical_patch_force_energy_against_radial_integral():
    R=.031;d=.03;k=1e7;rho=np.sqrt(R*R-d*d);a=exact_flat_sphere_patch(R,d,k)
    f=quad(lambda r:2*np.pi*r*k*(np.sqrt(r*r+d*d)-R)*d/np.sqrt(r*r+d*d),0,rho)[0]
    e=quad(lambda r:np.pi*r*k*(np.sqrt(r*r+d*d)-R)**2,0,rho)[0]
    np.testing.assert_allclose([a['force_normal_N'],a['energy_J']],[f,e],rtol=1e-12,atol=1e-15)
