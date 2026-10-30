import numpy as np
import pytest
from scripts.external_benchmark.fem import ExternalFEM
from scripts.contact_benchmark_v12.feasibility_verify import gradient_jacobians,topology,equal


def fixture():
    nodes=np.array([[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]],float)
    data=dict(nodes=nodes,controller_targets=nodes[None,None],attachment_weights=np.ones((1,4)),attachment_active=np.zeros((1,1)),floor=0.)
    return nodes,ExternalFEM(data,device='cpu')


def test_independent_rest_gradient_detects_inversion_and_preserves_rigid_pose():
    x,m=fixture();np.testing.assert_allclose(gradient_jacobians(m,x),1,atol=1e-12)
    y=x.copy();y[3,2]*=-1
    assert (gradient_jacobians(m,y)<0).all()
    from scipy.spatial.transform import Rotation
    R=Rotation.from_rotvec([.2,.3,-.1]).as_matrix()
    np.testing.assert_allclose(gradient_jacobians(m,x@R+[1.,2.,3.]),1,atol=1e-12)


def test_final_topology_checker_rejects_inversion_even_if_metadata_claims_valid():
    x,m=fixture();bad=x.copy();bad[3,2]*=-1
    record=dict(frames=1,failed_frames=0,frames_with_inversions=0,frames_below_solver_margin=0,per_frame=[dict(frame=0,failed=False,min_J=1,inverted_elements=0,below_solver_margin=0)])
    with pytest.raises(AssertionError,match='volume violation'):topology(m,bad[None],record)


def test_final_topology_checker_requires_failed_frame_metadata():
    x,m=fixture();record=dict(frames=1,failed_frames=1,frames_with_inversions=0,frames_below_solver_margin=0,per_frame=[dict(frame=0,failed=True)])
    result=topology(m,np.full_like(x[None],np.nan),record)
    assert result['failed_frames']==[0] and result['min_J'] is None
    record['per_frame'][0]['failed']=False
    with pytest.raises(AssertionError):topology(m,np.full_like(x[None],np.nan),record)


def test_nested_score_check_rejects_changed_error_or_missing_frame():
    original=dict(summary=dict(error=1.),frames=[dict(frame=3,error=2.)])
    equal(original,original)
    with pytest.raises(AssertionError):equal(original,dict(summary=dict(error=2.),frames=original['frames']))
    with pytest.raises(AssertionError):equal(original,dict(summary=original['summary'],frames=[]))
