import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu
import torch
from experiments.campaign_c38_static_family import build_geometry
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_consistent_mass_pcg import ConsistentMassPCG,MassSolveFailure


def test_actual_full_consistent_mass_LU_acceleration_kinetic_work_and_reactions():
 torch.set_num_threads(2)
 m=build_geometry((4,2,2),100000.,'both_ends',Fz=0.,rule='keast24');M=scalar_mass(m).tocsr();free=np.flatnonzero(~m.mask[:,0]);fixed=np.flatnonzero(m.mask[:,0]);A=M[free][:,free].tocsr();before=(A.data.copy(),A.indices.copy(),A.indptr.copy());solver=ConsistentMassPCG(A);lu=splu(A.tocsc())
 for b in [m.load[free],np.random.default_rng(581).normal(size=(len(free),3))*.01]:
  x=solver.solve(b);expected=lu.solve(b);record=solver.last_record
  assert record['passed'] and record['true_residual_N']<=1e-12
  assert np.linalg.norm(A@x-b)<=1e-12
  np.testing.assert_allclose(x,expected,atol=2e-10,rtol=2e-10)
  actual=np.zeros_like(m.rest);oracle=actual.copy();actual[free]=x;oracle[free]=expected
  np.testing.assert_allclose(M[fixed][:,free]@x,M[fixed][:,free]@expected,atol=2e-13,rtol=2e-10)
  kinetic=lambda v:.5*float(np.sum(v*(M@v)))
  assert abs(kinetic(actual)-kinetic(oracle))<=1e-9*max(1.,abs(kinetic(oracle)))
  # Same known force and manufactured displacement increment, no dynamics solve.
  assert abs(np.sum(m.load*(actual-oracle))*.001)<1e-12
  assert np.linalg.norm(x-b/A.diagonal()[:,None])>.01  # Full coupling is active.
 for old,new in zip(before,[A.data,A.indices,A.indptr]):np.testing.assert_array_equal(old,new)


def test_zero_RHS_and_hard_failures_never_return_uncertified_acceleration():
 A=sparse.csr_matrix(np.array([[2.,.7],[.7,1.]]));solver=ConsistentMassPCG(A)
 np.testing.assert_array_equal(solver.solve(np.zeros((2,3))),0)
 assert solver.last_record['iterations']==0 and solver.last_record['true_residual_N']==0
 bad=ConsistentMassPCG(sparse.csr_matrix(np.array([[1.,2.],[2.,1.]])))
 with pytest.raises(MassSolveFailure,match='curvature'):bad.solve(np.array([[1.,0,0],[-1.,0,0]]))
 assert not bad.last_record['passed']
 capped=ConsistentMassPCG(A,max_iterations=1)
 with pytest.raises(MassSolveFailure,match='iteration cap'):capped.solve(np.ones((2,3)))
 assert not capped.last_record['passed'] and capped.last_record['true_residual_N']>1e-12


def test_FP64_original_action_three_columns_and_strict_validation():
 A=sparse.csr_matrix(np.array([[3.,.4,0],[.4,2.,.2],[0,.2,1.]]));solver=ConsistentMassPCG(A,absolute_tolerance_N=1e-25)
 b=np.array([[1.,-2.,3.],[2.,1.,-1.],[-1.,3.,2.]])*1e-14;x=solver.solve(b)
 assert np.linalg.norm(A@x-b)<=1e-25
 np.testing.assert_allclose(solver.action(torch.tensor(x)).numpy(),A@x,atol=1e-29,rtol=1e-14)
 with pytest.raises(ValueError,match='FP64'):solver.solve(b.astype(np.float32))
 with pytest.raises(ValueError,match='Symmetric'):ConsistentMassPCG(sparse.csr_matrix(np.array([[1.,1.],[0,1.]])))
