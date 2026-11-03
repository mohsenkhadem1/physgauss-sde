import torch
from scripts.complex_scenes_v18.material_extension import IndependentBulkScale


def test_preserves_old_energy_stress_and_changes_only_compressibility():
    torch.manual_seed(91);law=IndependentBulkScale().double()
    F=(torch.eye(3,dtype=torch.float64)[None]+.1*torch.randn(12,3,3,dtype=torch.float64)).requires_grad_()
    z=torch.randn(12,4,dtype=torch.float64);z[:,3]=0
    old=law.base.energy(F,z[:,:3]);new=law.energy(F,z)
    assert torch.equal(old,new)
    a=torch.autograd.grad(old.sum(),F,retain_graph=True)[0];b=torch.autograd.grad(new.sum(),F)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Isochoric shear is independent of the new coordinate.
    shear=torch.eye(3,dtype=torch.float64)[None];shear[:,0,1]=.4
    base=torch.tensor([[0.,0.,-1.,0.]],dtype=torch.float64);changed=base.clone();changed[:,3]=1.
    torch.testing.assert_close(law.energy(shear,base),law.energy(shear,changed),rtol=0,atol=0)
    dilation=1.1*torch.eye(3,dtype=torch.float64)[None]
    assert law.energy(dilation,changed)>law.energy(dilation,base)
