import numpy as np
import torch
from scipy import sparse
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_mixed_cg import MixedCorrectionP2


def test_mixed_correction_matches_independent_direct_original_operator():
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1,order=3)
    t=MixedCorrectionP2(m);u=torch.zeros_like(t.rest);cache=t.evaluate(u);rng=np.random.default_rng(731);rhs=rng.normal(size=m.rest.shape);rhs[m.mask]=0
    for factor in [0.,1e6]:
        delta,_=t.cg(cache,torch.tensor(rhs),factor,rtol=1e-10,atol=1e-12)
        _,_,H,_=m.elastic(m.rest,True)
        M=sparse.csr_matrix((m.rest.size,m.rest.size))
        if factor:
            ids=np.arange(m.rest.size);basis=np.eye(m.rest.size).reshape(m.rest.size,*m.rest.shape)
            M=sparse.csr_matrix(np.stack([t.mass(torch.tensor(v)).numpy().ravel() for v in basis],axis=1))
        A=H+factor*M;expected=np.zeros(m.rest.size);expected[m.free]=sparse.linalg.spsolve(A[m.free][:,m.free],rhs.ravel()[m.free])
        assert np.max(abs(delta.numpy().ravel()-expected))<1e-10
        assert t.mixed_records[-1]['original_FP64_residual_N']<1e-8


def test_fp32_correction_failure_uses_original_fp64_fallback():
    m=StableP2Block((1,1,1),young=5000,gravity=0,traction=1,order=3);t=MixedCorrectionP2(m)
    from experiments.campaign_p2_torch import LinearFailure
    def fail(*args,**kwargs):raise LinearFailure('Deliberate inner failure')
    t.low.cg=fail
    a=t.solve()
    assert float(torch.linalg.vector_norm(a['residual']))<=1e-10
    assert all(r['fallback'] for r in t.mixed_records)
    assert abs(a['displacement'].numpy()[m.mask]).max()==0
