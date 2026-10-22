import numpy as np
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.diagnose_p2_trajectory_geometry import frame_bounds

def test_affine_field_bound_and_unsafe_field_do_not_claim_certificate():
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1,order=3)
    G=np.diag([.1,-.03,.02]);q=m.rest+m.rest@G.T+np.array([.1,-.2,.03])
    rows=frame_bounds(m.rest,m.tets,m.elements,np.stack([q,m.rest*3]))
    assert rows[0]['positive_whole_element_certificate']
    assert abs(rows[0]['sufficient_whole_element_minimum_J_bound']-.9**3)<1e-13
    assert rows[1]['positive_whole_element_certificate'] is False
    assert rows[1]['sufficient_whole_element_minimum_J_bound'] is None
