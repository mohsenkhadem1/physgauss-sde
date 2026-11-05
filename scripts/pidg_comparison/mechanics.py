"""Frozen shared constitutive energy transferred to a PIDG deformation field.
This is a reconstruction regularizer, not our full dynamics/contact solver.
"""
from pathlib import Path
import torch
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt'
if not CHECKPOINT.exists():
    CHECKPOINT=ROOT/'reports/pidg-comparison-v01/shared-material-law.pt'

def deformation_gradient(mapping,x,epsilon=.01):
    """Centered spatial finite differences; columns are d phi / d X_j."""
    delta=torch.eye(3,device=x.device,dtype=x.dtype)*epsilon
    plus=x[:,None,:]+delta;minus=x[:,None,:]-delta
    return ((mapping(plus.reshape(-1,3))-mapping(minus.reshape(-1,3))).reshape(-1,3,3)/(2*epsilon)).transpose(-1,-2)

class ConstitutivePrior:
    def __init__(self,device='cuda'):
        self.law=ScaleSeparatedMaterialLaw().to(device)
        self.law.load_state_dict(torch.load(CHECKPOINT,map_location=device,weights_only=False)['state_dict'])
        self.law.eval()
        for p in self.law.parameters():p.requires_grad_(False)
    def energy(self,F):
        J=torch.linalg.det(F)
        valid=J>.05
        # The learned law is defined for positive orientation only. An explicit
        # inversion penalty provides a separate gradient outside that domain.
        safe=torch.where(valid[:,None,None],F,torch.eye(3,device=F.device,dtype=F.dtype))
        z=F.new_tensor([0.,0.,-1.]).expand(len(F),3)
        W=self.law.energy(safe,z)/12000.
        return (W+10*torch.relu(.1-J).square()).mean()
    def __call__(self,deform,gaussians,fid):
        from utils.general_utils import quaternion_to_matrix
        # Deterministic samples and forked RNG leave baseline sampling unchanged.
        xyz=gaussians.get_xyz.detach()
        ids=torch.linspace(0,len(xyz)-1,min(256,len(xyz)),device=xyz.device).long()
        x=xyz[ids]
        def mapping(q):
            d=deform.step(q,fid.reshape(1,1).expand(len(q),1))['d_xyz']
            return (quaternion_to_matrix(d[:,:4])@q[...,None]).squeeze(-1)+d[:,4:]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            return self.energy(deformation_gradient(mapping,x))
