import torch
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow_v02.differential_action import rigid_residual,target_profile,rollout


def test_kabsch_removes_finite_rigid_motion():
    torch.manual_seed(72);x=torch.randn(2,5,12,3,dtype=torch.float64)
    R=x.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    y=x@R+x.new_tensor([.5,-2.,.1])
    assert rigid_residual(x,y).abs().max()<1e-12


def test_quadratic_profile_cannot_be_affine_rigid_motion():
    rest=torch.cartesian_prod(torch.linspace(-.05,.05,4,dtype=torch.float64),torch.tensor([.2,.3],dtype=torch.float64),torch.linspace(-.05,.05,4,dtype=torch.float64))
    A=torch.cat([rest,torch.ones(len(rest),1,dtype=rest.dtype)],-1)
    target=target_profile(rest)
    fit=A@torch.linalg.lstsq(A,target).solution
    assert (target-fit).square().mean().sqrt()>.1


def test_differential_command_preserves_prefix():
    b=RefinedBlock(n=2,dtype=torch.float64);z=torch.zeros(1,6,dtype=torch.float64)
    with torch.no_grad():
        first=b.rollout(z,kind='release');second=rollout(b,z,.003)
    assert torch.equal(first[:,:17],second[:,:17])
    assert (first[:,17:]-second[:,17:]).abs().max()>.0001
