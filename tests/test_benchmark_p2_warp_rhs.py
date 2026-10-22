import numpy as np
from experiments.benchmark_p2_warp_rhs import compare

def test_parity_rejects_force_energy_and_RHS_without_any_GPU():
 a=dict(energy=0.,minimum_J=1.,gradient=np.zeros((2,3)),acceleration=np.zeros((2,3)),masssolve_R_N=0.)
 assert compare(a,a,np.eye(2))['passed']
 b={**a,'gradient':np.full((2,3),1e-10)};assert not compare(a,b,np.eye(2))['passed']
 b={**a,'energy':2e-12};assert not compare(a,b,np.eye(2))['passed']
 b={**a,'acceleration':np.full((2,3),1e-10)};assert not compare(a,b,np.eye(2))['passed']
