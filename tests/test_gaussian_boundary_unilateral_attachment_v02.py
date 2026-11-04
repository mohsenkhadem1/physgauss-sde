import numpy as np
import pytest
import torch
from scripts.gaussian_boundary_flow_v02.unilateral_attachment import unilateral_plane,UnilateralAttachmentFEM
from scripts.external_benchmark.fem import ExternalFEM
D=torch.float64


def args():
    return dict(weights=torch.tensor([[.3,.7]],dtype=D),stiffness=torch.tensor([4.],dtype=D),
                active=torch.ones(1,dtype=D),origins=torch.zeros((1,3),dtype=D),
                normals=torch.tensor([[0.,0.,1.]],dtype=D))


def test_potential_gradient_hessian_and_compressive_force():
    x=torch.tensor([[.1,.2,-.2],[.2,-.1,.3]],dtype=D,requires_grad=True);a=args()
    e,r,b=unilateral_plane(x,**a);g,=torch.autograd.grad(e,x,create_graph=True)
    H=torch.autograd.functional.hessian(lambda q:unilateral_plane(q,**a)[0],x).reshape(6,6)
    assert torch.allclose(g,r)
    assert torch.allclose(H,torch.block_diag(*b))
    assert (-r[0,2])>0 and torch.count_nonzero(r[:, :2])==0


def test_separation_activation_and_tangential_freedom():
    a=args();x=torch.tensor([[0.,0.,-.1],[0.,0.,-.2]],dtype=D)
    e,r,b=unilateral_plane(x,**a)
    moved=x+torch.tensor([2.,-3.,0.],dtype=D)
    e2,r2,b2=unilateral_plane(moved,**a)
    assert torch.equal(e,e2) and torch.equal(r,r2) and torch.equal(b,b2)
    for q,active in [(x+torch.tensor([0,0,1.],dtype=D),torch.ones(1,dtype=D)),(x,torch.zeros(1,dtype=D))]:
        a['active']=active
        assert all(torch.count_nonzero(v)==0 for v in unilateral_plane(q,**a))
    # Withdrawal moves the plane away from the object: exactly zero tensile force.
    a=args();a['origins'][:,2]=-1
    assert torch.count_nonzero(unilateral_plane(x,**a)[1])==0


def test_task_frame_equivariance():
    from scipy.spatial.transform import Rotation
    a=args();x=torch.tensor([[.1,.2,-.2],[.2,-.1,.3]],dtype=D)
    R=torch.tensor(Rotation.from_rotvec([.3,-.2,.4]).as_matrix(),dtype=D);shift=torch.tensor([1.,-2.,3.],dtype=D)
    e,r,H=unilateral_plane(x,**a)
    a['origins']=a['origins']@R+shift;a['normals']=a['normals']@R
    e2,r2,H2=unilateral_plane(x@R+shift,**a)
    assert torch.allclose(e,e2) and torch.allclose(r2,r@R)
    assert torch.allclose(H2,R.T@H@R)


def fixture():
    nodes=np.array([[0.,0.,0.],[1.,0,0],[0,1.,0],[0,0,1.]])*.1
    return dict(nodes=nodes,attachment_weights=np.array([[1.,.5,.5,.2]]),
                controller_targets=nodes[None,None]+np.array([0.,0.,-.01]),
                attachment_active=np.ones((1,1)),floor=-.1)


def test_fem_nested_bilateral_and_unilateral_derivatives():
    data=fixture();plane=np.array([[[0.,0.,.02]]]);normal=np.array([[[0.,0.,1.]]])
    base=ExternalFEM(data,device='cpu',field=False)
    theta=torch.zeros(base.i_field+1,dtype=D)
    nested=UnilateralAttachmentFEM(data,device='cpu',field=False,plane_origins=plane,plane_normals=normal,unilateral_groups=[False])
    for a,b in zip(base.evaluate(base.rest,theta,0,True),nested.evaluate(nested.rest,theta,0,True)):
        assert torch.equal(a,b)
    model=UnilateralAttachmentFEM(data,device='cpu',field=False,plane_origins=plane,plane_normals=normal,unilateral_groups=[True])
    x=model.rest.clone().requires_grad_(True)
    e,r,H=model.evaluate(x,theta,0,True);gradient,=torch.autograd.grad(e,x)
    autoh=torch.autograd.functional.hessian(lambda q:model.evaluate(q,theta,0)[0],x).reshape(model.ndof,model.ndof)
    assert torch.allclose(gradient,r,atol=1e-10)
    assert torch.allclose(autoh,H,atol=1e-10)


def test_reject_bad_normal_and_nan():
    a=args();a['normals']*=2
    with pytest.raises(ValueError,match='unit'):unilateral_plane(torch.zeros((2,3),dtype=D),**a)
    a=args();a['stiffness'][:]=float('nan')
    with pytest.raises(ValueError,match='Finite'):unilateral_plane(torch.zeros((2,3),dtype=D),**a)
