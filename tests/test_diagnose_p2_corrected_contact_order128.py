import numpy as np
from experiments.diagnose_p2_corrected_contact_order128 import compare

def test_strict_force_energy_components_reject_known32_error():
 a=dict(energy=1.,contact_energy=.1,gradient=np.zeros((2,3)));assert compare(a,a)['passed']
 assert not compare(a,{**a,'gradient':np.full((2,3),1e-8)})['passed']
 assert not compare(a,{**a,'energy':1+2e-12})['passed']
 assert not compare(a,{**a,'contact_energy':.1+2e-12})['passed']
