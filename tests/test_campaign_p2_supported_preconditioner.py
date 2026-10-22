import numpy as np
import pytest
import torch
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2
from experiments.campaign_p2_supported_preconditioner import SupportedTwoLevelP2, StreamingSupportedP2


def test_original_fullleft_backend_and_new_streamed_branch_parity():
    torch.set_num_threads(2)
    m=P2Geometry((2,2,2),pattern='axial_pull')
    a=TwoLevelP2(m);b=SupportedTwoLevelP2(m,support_kind='left_face')
    c=StreamingSupportedP2(m,support_kind='left_face',quadrature_chunk_size=13)
    torch.testing.assert_close(a.Kc,b.Kc,atol=0,rtol=0)
    torch.testing.assert_close(a.Kc,c.Kc,atol=0,rtol=0)
    sa=a.solve();sb=b.solve();sc=c.solve()
    torch.testing.assert_close(sa['positions'],sb['positions'],atol=2e-12,rtol=0)
    torch.testing.assert_close(sa['positions'],sc['positions'],atol=2e-12,rtol=0)
    assert not hasattr(c,'grad') and len(c.bary)==27


@pytest.mark.parametrize('kind',['right_face','both_ends','bottom_face','separated_bottom_patches'])
def test_fixed_support_operator_mass_and_original_linear_residual(kind):
    m=P2Geometry((4,2,2),support=kind,gravity=9.81,young=100000.)
    mask=m.mask.copy();load=m.load.copy()
    a=StreamingSupportedP2(m,support_kind=kind,quadrature_chunk_size=13)
    b=TorchP2(m)
    rng=np.random.default_rng(72)
    u=torch.tensor(rng.normal(0,1e-5,m.rest.shape),dtype=torch.float64);u[a.mask]=0
    v=torch.tensor(rng.normal(size=m.rest.shape),dtype=torch.float64);v[a.mask]=0
    ca=a.evaluate(u);cb=b.evaluate(u)
    torch.testing.assert_close(ca['gradient'],cb['gradient'],atol=1e-10,rtol=2e-12)
    torch.testing.assert_close(a.tangent(ca,v),b.tangent(cb,v),atol=2e-9,rtol=2e-12)
    torch.testing.assert_close(a.mass(v),b.mass(v),atol=0,rtol=0)
    x,k=a.cg(ca,v,1000.,rtol=1e-11,atol=1e-12)
    residual=(b.tangent(cb,x)+1000*b.mass(x)-v)[~a.mask]
    assert float(torch.linalg.vector_norm(residual))<1e-8
    np.testing.assert_array_equal(x.numpy()[mask],0)
    np.testing.assert_array_equal(m.mask,mask);np.testing.assert_array_equal(m.load,load)
    # New support branch does not broaden the old constructor's declaration.
    with pytest.raises(ValueError,match='full-left'):
        TwoLevelP2(m)


def test_support_declaration_mismatch_and_unconstrained_model_rejected():
    m=P2Geometry((2,2,2))
    with pytest.raises(ValueError,match='Explicit'):
        SupportedTwoLevelP2(m)
    with pytest.raises(ValueError,match='exactly'):
        StreamingSupportedP2(m,support_kind='bottom_face')
    m.mask[:]=False
    with pytest.raises(ValueError,match='exactly'):
        SupportedTwoLevelP2(m,support_kind='left_face')
