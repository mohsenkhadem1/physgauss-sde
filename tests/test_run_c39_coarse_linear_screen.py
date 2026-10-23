import numpy as np
from scipy import sparse
from experiments.run_c39_coarse_linear_screen import constrained_solve,fisher,ACTIONS,CANDIDATES

def test_exact_mask_elimination_preserves_constraint_load_reactions():
 K=sparse.csr_matrix([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]])
 force=np.array([[99.,2.,3.]]);mask=np.array([[True,False,False]])
 u,R=constrained_solve(K,force,mask)
 np.testing.assert_allclose(u[0,1:],np.linalg.solve(K.toarray()[1:,1:],force[0,1:]),rtol=1e-15)
 assert u[0,0]==0 and R<1e-14

def test_logE_fisher_uses_total_displacement_and_fixed_noise():
 u=np.arange(27).reshape(9,3)*1e-5
 assert fisher(2*u)==4*fisher(u)
 assert fisher(u)==np.sum(u*u)/.0002**2
 assert ACTIONS==('gravity',)+CANDIDATES+('canonical_up_1N',)
