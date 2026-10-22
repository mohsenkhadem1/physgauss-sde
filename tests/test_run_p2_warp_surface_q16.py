import numpy as np
from experiments.run_p2_warp_surface_q16 import surface_pair,CONFIG,manual_dop853
from experiments.continue_p2_contact_dop853_v03 import manual_dop853 as original

def test_same_tight_time_method_and_fixed_surface_gates():
 assert manual_dop853 is original
 assert CONFIG==dict(name='surface_Q16',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
 q=np.zeros((9,225,3));v=q.copy();E=np.zeros(9)
 assert surface_pair(q,v,E,q,v,E,4.7088e-7)['passed']
 changed=q.copy();changed[3,4,2]=21e-6;assert not surface_pair(q,v,E,changed,v,E,4.7088e-7)['passed']
 changed=v.copy();changed[4,6,0]=1.1e-4;assert not surface_pair(q,v,E,q,changed,E,4.7088e-7)['passed']
 changed=E.copy();changed[5]=5e-7;assert not surface_pair(q,v,E,q,v,changed,4.7088e-7)['passed']
