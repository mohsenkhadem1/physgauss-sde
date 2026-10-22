import numpy as np
from experiments.run_p2_warp_96_bridge_v02 import bridge_pair,CONFIG,manual_dop853
from experiments.continue_p2_contact_dop853_v03 import manual_dop853 as original_manual

def test_strict_bridge_gates_and_original_step_helper():
 assert manual_dop853 is original_manual
 assert CONFIG==dict(name='warp_bridge',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
 q=np.zeros((9,225,3));v=q.copy();E=np.zeros(9)
 assert bridge_pair(q,v,E,q,v,E,4.7088e-7)['passed']
 delta=q.copy();delta[0,2,1]=1.1e-8;assert not bridge_pair(q,v,E,delta,v,E,4.7088e-7)['passed']
 delta=v.copy();delta[1,4,2]=1.1e-5;assert not bridge_pair(q,v,E,q,delta,E,4.7088e-7)['passed']
 delta=E.copy();delta[3]=5e-7;assert not bridge_pair(q,v,E,q,v,delta,4.7088e-7)['passed']


def test_full_prefix_reserve_accounts_for_states_and_outputs():
 from experiments.run_p2_warp_96_bridge_v02 import estimate_prefix_bytes
 n,outputs,d=25000,9,1353
 assert estimate_prefix_bytes(n,outputs,d)>(n+outputs+2)*d*8
