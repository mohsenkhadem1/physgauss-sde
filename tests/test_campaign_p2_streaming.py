import numpy as np
import pytest
import torch
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
from experiments.campaign_p2_streaming import StreamingTwoLevelP2


def model():return ExactBulkP2Geometry((1,1,1),young=4500,pattern='axial_pull')


def state(m):
    X=m.rest;u=np.column_stack([.04*X[:,0]+.3*X[:,1]*X[:,2],-.02*X[:,1]+.1*X[:,0]*X[:,2],.03*X[:,2]+.2*X[:,0]*X[:,1]])
    return torch.tensor(u,dtype=torch.float64)


def test_constructor_never_mutates_input_and_no_full_quad_cache():
    torch.set_num_threads(2);m=model();b=m.bary.copy();w=m.weights.copy();checks=m.check_bary.copy();t=StreamingTwoLevelP2(m,quadrature_chunk_size=8)
    np.testing.assert_array_equal(m.bary,b);np.testing.assert_array_equal(m.weights,w);np.testing.assert_array_equal(m.check_bary,checks)
    assert len(t.bary)==80 and len(t.check_bary)==10 and not any(hasattr(t,k) for k in ['grad','check_grad','weights'])
    a=t.evaluate(state(m));assert set(a)=={'local_u','energy','gradient','minimum_J'}
    assert a['local_u'].shape==(6,10,3)
    for k,v in a.items():assert not (v.ndim>=2 and v.shape[:2]==(80,6))


@pytest.mark.parametrize('chunk',[8,13])
def test_exact80_energy_gradient_fullHVP_nodalblocks_and_diagonal_parity(chunk):
    m=model();a=StreamingTwoLevelP2(m,quadrature_chunk_size=chunk);b=BlockJacobiP2(m);u=state(m);v=torch.tensor(np.random.default_rng(19).normal(size=m.rest.shape));ca=a.evaluate(u);cb=b.evaluate(u)
    torch.testing.assert_close(ca['energy'],cb['energy'],atol=2e-14,rtol=2e-13)
    torch.testing.assert_close(ca['gradient'],cb['gradient'],atol=2e-11,rtol=2e-12)
    torch.testing.assert_close(a.tangent(ca,v),b.tangent(cb,v),atol=2e-9,rtol=2e-12)
    torch.testing.assert_close(a.nodal_blocks(ca,17.),b.nodal_blocks(cb,17.),atol=2e-10,rtol=2e-12)
    torch.testing.assert_close(a.diagonal(ca),b.diagonal(cb),atol=2e-10,rtol=2e-12)
    torch.testing.assert_close(a.mass(v),b.mass(v),atol=0,rtol=0)


def test_full_consistent_mass_linear_solve_and_original_static_parity():
    m=model();a=StreamingTwoLevelP2(m,quadrature_chunk_size=13);b=TorchP2(m);u=state(m);u[a.mask]=0;ca=a.evaluate(u);cb=b.evaluate(u);rhs=torch.tensor(np.random.default_rng(22).normal(size=m.rest.shape));rhs[a.mask]=0
    x,k=a.cg(ca,rhs,1000.,rtol=1e-11,atol=1e-12);y,j=b.cg(cb,rhs,1000.,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(x,y,atol=2e-10,rtol=2e-9)
    actual=(a.tangent(ca,x)+1000*a.mass(x)-rhs)[~a.mask];assert float(torch.linalg.vector_norm(actual))<1e-9
    sa=a.solve();sb=b.solve();torch.testing.assert_close(sa['positions'],sb['positions'],atol=2e-10,rtol=0)
    assert float(torch.linalg.vector_norm(sa['residual']))<=1e-10
    np.testing.assert_array_equal(sa['displacement'].numpy()[m.mask],0.)


def test_nonpositiveJ_rejected_without_clipping_and_chunk_validation():
    m=model();a=StreamingTwoLevelP2(m,quadrature_chunk_size=13);u=torch.zeros_like(a.rest);u[:,0]=-2*a.rest[:,0]
    with pytest.raises(ValueError,match='Jacobian'):a.evaluate(u)
    with pytest.raises(ValueError,match='chunk'):StreamingTwoLevelP2(m,quadrature_chunk_size=0)
