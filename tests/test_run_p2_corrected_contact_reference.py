import numpy as np
from experiments.run_p2_corrected_contact_reference import pair_metrics,CONFIGS,manual_dop853
from experiments.continue_p2_contact_dop853_v03 import manual_dop853 as original

def test_frozen_pair_clock_method_configs_and_unchanged_gates():
 assert manual_dop853 is original
 assert [(c['rtol'],c['displacement_atol_m'],c['velocity_atol_m_s']) for c in CONFIGS]==[(1e-9,1e-12,1e-10),(1e-10,1e-13,1e-11)]
 q=np.zeros((9,225,3));v=q.copy();E=np.zeros(9);P=np.zeros((9,81,3));assert pair_metrics(q,v,E,q,v,E,4.7088e-7,P,P)['passed']
 qq=q.copy();qq[1,3,2]=21e-6;assert not pair_metrics(q,v,E,qq,v,E,4.7088e-7,P,P)['passed']
 vv=v.copy();vv[1,3,2]=1.1e-4;assert not pair_metrics(q,v,E,q,vv,E,4.7088e-7,P,P)['passed']
 EE=E.copy();EE[4]=5e-7;assert not pair_metrics(q,v,E,q,v,EE,4.7088e-7,P,P)['passed']
 PP=P.copy();PP[4,3,1]=21e-6;assert not pair_metrics(q,v,E,q,v,E,4.7088e-7,P,PP)['passed']
