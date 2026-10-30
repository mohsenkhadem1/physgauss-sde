import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.contact_benchmark_v12.feasibility import repair,jacobians


def fixture():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])*.04
    elements=np.array([[0,1,2,3],[1,2,3,4]])
    base=np.broadcast_to(nodes,(4,5,3)).copy();proposal=base.copy();proposal[:,3,2]*=-.5
    return dict(nodes=nodes,times=np.arange(4)*.1),elements,base,proposal


def test_every_inherited_tet_passes_and_anchor_pose_retained():
    d,e,b,p=fixture();out,meta=repair(d,b,p,elements=e)
    assert not meta['failed_frames'] and len(meta['modified_frames'])==4
    for frame in out:assert (jacobians(d['nodes'],frame,e)>.03+1e-8).all()
    assert all(r['alpha']<1 for r in meta['frames'])


def test_guard_is_rigid_equivariant_and_preserves_inputs():
    d,e,b,p=fixture();before=p.copy();out,meta=repair(d,b,p,elements=e)
    R=Rotation.from_rotvec([.2,-.4,.1]).as_matrix();v=np.array([.2,-.1,.5]);trans=dict(d,nodes=d['nodes']@R+v)
    changed,other=repair(trans,b@R+v,p@R+v,elements=e)
    np.testing.assert_allclose(changed,out@R+v,atol=1e-12)
    assert [r['alpha'] for r in meta['frames']]==[r['alpha'] for r in other['frames']]
    np.testing.assert_array_equal(p,before)


def test_feasible_proposal_is_unchanged():
    d,e,b,p=fixture();p=b+np.array([.1,.2,.3]);out,meta=repair(d,b,p,elements=e)
    np.testing.assert_array_equal(out,p);assert not meta['modified_frames']


def test_bad_anchor_and_nonfinite_proposal_remain_explicit_failures():
    d,e,b,p=fixture();b[0,3,2]*=-1;p[1]=np.nan
    out,meta=repair(d,b,p,elements=e)
    assert meta['failed_frames']==[0,1];assert np.isnan(out[:2]).all()


def test_guard_never_accepts_evaluation_payload():
    d,e,b,p=fixture();d['evaluation']={'points':p}
    with pytest.raises(ValueError,match='Evaluator truth'):repair(d,b,p,elements=e)


def test_unused_observations_do_not_change_guard():
    d,e,b,p=fixture();a,meta=repair(d,b,p,elements=e)
    d.update(observed=np.full((100,5,3),np.nan),observed_mask=np.zeros((100,5),bool))
    z,other=repair(d,b,p,elements=e);np.testing.assert_array_equal(a,z)


def test_jacobians_match_fem_reference_gradients():
    from scripts.external_benchmark.fem import ExternalFEM
    d,e,b,p=fixture();d.update(controller_targets=b[:,None],attachment_weights=np.ones((1,5)),attachment_active=np.ones((4,1)),floor=0.)
    model=ExternalFEM(d,device='cpu');elements=model.elements.numpy();x=model.t((p[0]-model.origin)/model.scale)
    import torch
    J=torch.linalg.det(torch.einsum('eai,eaj->eij',x[model.elements],model.grad)).numpy()
    np.testing.assert_allclose(jacobians(d['nodes'],p[0],elements),J,atol=1e-12)
