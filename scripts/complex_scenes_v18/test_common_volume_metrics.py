import numpy as np
from scripts.complex_scenes_v18.common_volume_metrics import ratios


def test_shared_volume_detects_reflection_and_preserves_affine_determinant():
    k=5;q=np.stack(np.meshgrid(*[np.linspace(0,1,k)]*3,indexing='ij'),-1).reshape(-1,3);rest=q*[.12,.06,.06]
    F=np.array([[.7,.2,0],[0,1.2,.1],[0,0,1.1]])
    x=(rest@F.T+[.1,.2,.3])[None,None]
    np.testing.assert_allclose(ratios(x,k),np.linalg.det(F),atol=1e-12)
    x[...,0]*=-1
    assert (ratios(x,k)<0).all()
