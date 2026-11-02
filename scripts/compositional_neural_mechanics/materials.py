"""Shared objective isotropic bulk energy with constrained latent conditioning.

For J>0 the isochoric invariants q1,q2 are nonnegative mathematically. Positive
coefficients preserve energy nonnegativity and exact reference normalization
for every latent. This is NOT a proof of polyconvexity, rank-one convexity, or
stable global equilibria. No anisotropic or history-dependent response is modeled.
All operations are pointwise; there is no batch normalization or sample pooling.
"""
import torch
from torch import nn
from torch.nn import functional as functional

def invariants(F):
    if F.shape[-2:]!=(3,3):raise ValueError('F must have final dimensions3x3')
    J=torch.linalg.det(F)
    if not bool(torch.isfinite(F).all())or not bool((J>0).all()):raise ValueError('Finite orientation-preserving F required')
    C=F.transpose(-1,-2)@F
    I1=C.diagonal(dim1=-2,dim2=-1).sum(-1)
    I2=.5*(I1.square()-(C*C).sum((-2,-1)))
    q1=J.pow(-2/3)*I1-3
    q2=J.pow(-4/3)*I2-3
    volume=J-torch.log(J)-1
    return q1,q2,volume

class SharedMaterialLaw(nn.Module):
    """Global shared weights; scene z is an explicit input, not a per-scene model.

Latent MLP may produce arbitrary logits, but only positive coefficients multiply
fixed admissible bases. Learned scalar basis slopes remain positive too. Energy
units are Pa, i.e. J/m³. The reference state has W=0 and first Piola stressP=0.
"""
    def __init__(self,latent_dim=2,hidden=16,energy_scale=12000.):
        super().__init__()
        if energy_scale<=0:raise ValueError('Positive energy scale required')
        self.latent_dim=latent_dim;self.energy_scale=float(energy_scale)
        self.coefficient_net=nn.Sequential(nn.Linear(latent_dim,hidden),nn.SiLU(),nn.Linear(hidden,7))
        self.raw_slopes=nn.Parameter(torch.tensor([0.,1.]))
        nn.init.zeros_(self.coefficient_net[-1].weight)
        nn.init.constant_(self.coefficient_net[-1].bias,-2.)
    def coefficients(self,z):
        if z.shape[-1]!=self.latent_dim:raise ValueError('Wrong latent dimension')
        return functional.softplus(self.coefficient_net(z))+1e-8
    def energy_density(self,F,z):
        q1,q2,volume=invariants(F);c=self.coefficients(z)
        slopes=functional.softplus(self.raw_slopes)+1e-4
        q=q1[...,None]*slopes
        # Convex Bregman residual of softplus about0: value and derivative0 at0.
        residual=functional.softplus(q)-functional.softplus(torch.zeros_like(q))-.5*q
        bases=torch.stack((.5*q1,.5*q2,volume,q1.square(),q1.pow(3),residual[...,0],residual[...,1]),dim=-1)
        return self.energy_scale*(c*bases).sum(-1)
    forward=energy_density
    energy=energy_density
    def stress(self,F,z,create_graph=True):
        with torch.enable_grad():
            if not F.requires_grad:F=F.detach().clone().requires_grad_(True)
            return torch.autograd.grad(self.energy_density(F,z).sum(),F,create_graph=create_graph)[0]

def teacher_energy(F,z,family):
    """Oracle constitutive generator only; never an observation/inference input.

z[...,0] is log scale (mu=4000exp(z0), bulk=20000exp(z0));
z[...,1] modulates family shape through sigmoid. MR partitionsmu betweenI1/I2;
Yeoh adds q1²/q1³ strain-stiffening. Both families share the same small-strain
shear modulus and bulk modulus at a given z, so predictingE alone cannot fit both.
"""
    if z.shape[-1]!=2:raise ValueError('Teacher latent dimension2 required')
    q1,q2,volume=invariants(F);mu=4000*torch.exp(z[...,0]);bulk=20000*torch.exp(z[...,0]);shape=torch.sigmoid(z[...,1])
    if family=='mooney_rivlin':
        fraction=.15+.6*shape
        shear=.5*mu*((1-fraction)*q1+fraction*q2)
    elif family=='yeoh':shear=mu*(.5*q1+(.08+.3*shape)*q1.square()+(.01+.08*shape)*q1.pow(3))
    else:raise ValueError('Unknown teacher family')
    return shear+bulk*volume
