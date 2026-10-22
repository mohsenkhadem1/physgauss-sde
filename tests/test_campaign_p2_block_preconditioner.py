import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2


def test_exact_nodal_block_matches_full_cross_coupled_Hessian():
    torch.set_num_threads(2);m=StableP2Block((1,1,1),young=6000,gravity=0,traction=1)
    t=BlockJacobiP2(m);u=np.random.default_rng(31).normal(size=m.rest.shape)*.0001;u[m.mask]=0
    cache=t.evaluate(torch.tensor(u));factor=7.3;blocks=t.nodal_blocks(cache,factor).numpy();H=m.elastic(m.rest+u,True)[2].toarray()
    expected=np.stack([H[3*i:3*i+3,3*i:3*i+3] for i in range(len(u))])+factor*t.mass_diagonal[:,0].numpy()[:,None,None]*np.eye(3)
    np.testing.assert_allclose(blocks,expected,atol=3e-12,rtol=1e-12);assert np.linalg.eigvalsh(blocks).min()>0


def test_original_linear_operator_true_residual_and_direct_solve_parity():
    m=StableP2Block((1,1,1),young=6000,gravity=0,traction=1);t=BlockJacobiP2(m);u=torch.zeros_like(t.rest);cache=t.evaluate(u);factor=250.;rhs=torch.tensor(np.random.default_rng(3).normal(size=m.rest.shape));rhs[t.mask]=0
    x,k=t.cg(cache,rhs,factor,rtol=1e-11,atol=1e-12)
    H=m.elastic(m.rest,True)[2]+factor*sparse.kron(sparse.csr_matrix(np.zeros((len(u),len(u)))),sparse.eye(3))
    # Obtain full M by independent order4 quadrature of shape functions.
    from experiments.campaign_static_p2_block import duffy,shapes
    b,w=duffy(4);N=shapes(b);M=np.zeros((len(u),len(u)))
    for T,V in zip(m.elements,m.volumes):M[np.ix_(T,T)]+=m.density*V*np.einsum('q,qa,qb->ab',w,N,N)
    H=m.elastic(m.rest,True)[2]+factor*sparse.kron(sparse.csr_matrix(M),sparse.eye(3));free=m.free;expected=spsolve(H[free][:,free],rhs.numpy().ravel()[free]);np.testing.assert_allclose(x.numpy().ravel()[free],expected,atol=2e-12,rtol=1e-9)
    actual=(t.tangent(cache,x)+factor*t.mass(x)-rhs)[~t.mask];assert torch.linalg.vector_norm(actual)<1e-9
    assert t.linear_records[-1]['regularization']==0 and k<=4000


def test_nonlinear_same_objective_force_stationarity_and_exact_clamp():
    m=StableP2Block((1,1,1),young=6000,gravity=0,traction=1);t=BlockJacobiP2(m);answer=t.solve();reference=m.solve()
    np.testing.assert_allclose(answer['positions'].numpy(),reference['positions'],atol=2e-11,rtol=0)
    assert torch.linalg.vector_norm(answer['residual'])<=1e-10
    np.testing.assert_array_equal(answer['displacement'].numpy()[m.mask],0.)
    assert all(x['regularization']==0 for x in t.linear_records)
