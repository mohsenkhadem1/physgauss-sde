import numpy as np
from experiments.run_p2_warp_trajectory_tight import parity,CONFIG

def test_strict_implementation_thresholds_and_exact_config():
 q=np.zeros((9,27,3));v=q.copy();assert parity(q,v,q,v)['passed']
 d=q.copy();d[3,4,0]=2e-8;assert not parity(q,v,d,v)['passed']
 d=v.copy();d[4,6,2]=2e-5;assert not parity(q,v,q,d)['passed']
 assert CONFIG==dict(rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
