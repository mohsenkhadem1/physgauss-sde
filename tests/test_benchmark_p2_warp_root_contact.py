import numpy as np
from experiments.benchmark_p2_warp_root_contact import parity,GPU

def test_unchanged_strict_E_g_gates_and_physicalGPU1():
 a=dict(contact_energy=1.,gradient=np.zeros((3,3)));g=dict(energy_J=1e-12,gradient_L2_N=1e-10)
 assert parity(a,a,g)['passed']
 assert not parity(a,{**a,'contact_energy':1+2e-12},g)['passed']
 assert not parity(a,{**a,'gradient':np.full((3,3),1e-10)},g)['passed']
 assert GPU=='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb'
