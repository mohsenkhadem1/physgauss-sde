import numpy as np
from scripts.compositional_neural_mechanics.contact_gap_search import map_jacobian,search_gaps
from scripts.compositional_neural_mechanics.contact_resolution import ResolvedContactQ2
from scripts.compositional_neural_mechanics.contact_mechanics import RigidSphere
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec


def test_map_derivative_and_continuous_known_sphere_gap():
    g=GeometrySpec(taper=.2,arch=.01);q=np.array([.3,.4,.6]);h=1e-6
    fd=np.column_stack([(g.map(q+h*np.eye(3)[i])-g.map(q-h*np.eye(3)[i]))/(2*h)for i in range(3)])
    np.testing.assert_allclose(map_jacobian(g,q),fd,atol=1e-10)
    m=ResolvedContactQ2(n=1,colliders=(RigidSphere(center=(.123,.003,-.07),radius=.031),));result=search_gaps(m,np.zeros_like(m.rest),grid=17,seeds_per_face=2)[0]
    assert abs(result['max_penetration_m']-.001)<1e-10
    assert result['dense_underestimate_m']>0
