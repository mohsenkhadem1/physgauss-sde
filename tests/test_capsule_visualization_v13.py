import numpy as np
from scipy.spatial.transform import Rotation
from scripts.coupled_contact_v13.capsule_development import CapsuleCollider
from scripts.coupled_contact_v13.capsule_visualization import capsule_bounds,capsule_surfaces


def test_capsule_drawing_matches_sdf_and_conservative_world_bounds():
    c=CapsuleCollider(center=(.01,0,.042),radius=.008,rotation=Rotation.from_rotvec([.2,.3,.4]).as_matrix())
    points=np.concatenate([np.moveaxis(a,0,-1).reshape(-1,3) for a in capsule_surfaces(c)])
    np.testing.assert_allclose(c.query(points)[0],0,atol=1e-14)
    lo,hi=capsule_bounds(c)
    assert np.all(points>=lo-1e-15) and np.all(points<=hi+1e-15)
