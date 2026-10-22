import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2


def test_exact_nontrivial_prolongation_constant_affine_and_fixed_compatible_fields():
 torch.set_num_threads(2);m=StableP2Block((8,4,4),young=4500,gravity=0,traction=1);t=TwoLevelP2(m);P=t.prolongation_csr;X=t.coarse_rest.numpy()
 np.testing.assert_allclose(P@np.ones(len(X)),1,atol=3e-15)
 np.testing.assert_allclose(P@X,m.rest,atol=3e-16)
 a=X[:,0,None]*np.array([[.02,-.01,.03]]);pred=t.prolong(torch.tensor(a.ravel()[t.coarse_free.numpy()])).numpy()
 np.testing.assert_allclose(pred,m.rest[:,0,None]*np.array([[.02,-.01,.03]]),atol=2e-17)
 np.testing.assert_array_equal(pred[m.mask],0.)


def test_projected_prolongation_restriction_adjoint_and_SPD_additive_operator():
 m=StableP2Block((4,2,2),young=4500,gravity=0,traction=1);t=TwoLevelP2(m);rng=np.random.default_rng(6);r=torch.tensor(rng.normal(size=m.rest.shape));s=torch.tensor(rng.normal(size=m.rest.shape));c=torch.tensor(rng.normal(size=len(m.free)))
 torch.testing.assert_close((r*t.prolong(c)).sum(),(t.restrict(r)*c).sum(),atol=2e-12,rtol=1e-12)
 cache=t.evaluate(torch.zeros_like(t.rest));B=t.make_preconditioner(cache,37.);Br=B(r);Bs=B(s)
 assert float((r*Br).sum())>0
 torch.testing.assert_close((r*Bs).sum(),(s*Br).sum(),atol=2e-11,rtol=2e-12)
 np.testing.assert_array_equal(Br.numpy()[m.mask],0.)
 assert torch.linalg.cholesky_ex(t.Kc+37*t.Mc)[1]==0


def test_original_FP64_linear_solve_parity_with_full_sparse_direct_reference():
 m=StableP2Block((4,2,2),young=4500,gravity=0,traction=1);t=TwoLevelP2(m);u=np.random.default_rng(9).normal(size=m.rest.shape)*1e-5;u[m.mask]=0.;cache=t.evaluate(torch.tensor(u));rhs=torch.tensor(np.random.default_rng(10).normal(size=m.rest.shape));rhs[t.mask]=0;mf=21.;x,k=t.cg(cache,rhs,mf,rtol=1e-12,atol=1e-12)
 H=m.elastic(m.rest+u,True)[2]+mf*sparse.kron(scalar_mass(m),sparse.eye(3));free=m.free;reference=spsolve(H[free][:,free],rhs.numpy().ravel()[free]);np.testing.assert_allclose(x.numpy().ravel()[free],reference,atol=1e-10,rtol=1e-10)
 original=(t.tangent(cache,x)+mf*t.mass(x)-rhs)[~t.mask];assert float(torch.linalg.vector_norm(original))<=1e-9
 assert all(r['regularization']==0 for r in t.linear_records) and k<=4000


def test_scope_rejects_incompatible_coarse_constraint_semantics():
 m=StableP2Block((1,1,1),young=4500,gravity=0,traction=1);m.mask[:]=False
 with pytest.raises(ValueError,match='full-left'):TwoLevelP2(m)
