"""Frozen constitutive decoder on arbitrary tetrahedral geometry.

Shares the external FEM's explicit tool attachments, gravity and floor. This is
an external tetrahedral adapter, not the paper's structured Q2 discretization.
No future shape, true force or material label is accepted by the predictor.
"""
from pathlib import Path
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw

ROOT = Path(__file__).resolve().parents[2]
BULK = ROOT / 'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt'


class NeuralFEM(ExternalFEM):
    def __init__(self, data, device='cuda:0', materials=6):
        if 'evaluation' in data:
            raise ValueError('Evaluator data forbidden')
        super().__init__(data, device=device, field=False, contact=False,
                         nonconvex=True, materials=materials)
        self.law = ScaleSeparatedMaterialLaw().to(device=device, dtype=self.dtype)
        self.law.load_state_dict(torch.load(BULK, map_location=device, weights_only=False)['state_dict'])
        self.law.requires_grad_(False)
        self.extra = self.i_field + 1

    def bulk(self, x, theta, tangent=False):
        # Explicit enable_grad also supports the parent's no_grad Newton solver.
        with torch.enable_grad():
            F = torch.einsum('eai,eaj->eij', x[self.elements], self.grad)
            if not F.requires_grad:
                F = F.detach().requires_grad_(True)
            J = torch.linalg.det(F)
            if float(J.detach().min()) <= .03:
                raise ValueError('Inverted or nearly singular element')
            z = torch.stack([self.feature @ theta[self.i_material:self.i_field],
                             theta[self.extra].expand(len(F)),
                             theta[self.extra+1].expand(len(F))], -1)
            W = self.law.energy(F, z) / self.law.base.energy_scale
            c = self.law.base.coefficients(z[..., 1:])[..., 2]
            W = W + torch.expm1(theta[self.extra+2]) * torch.exp(z[..., 0]) * c * (J-torch.log(J)-1)
            P, = torch.autograd.grad(W.sum(), F, create_graph=True)
            H = None
            if tangent:
                # Elements are independent: nine summed VJPs give all local 9x9 tangents.
                rows = [torch.autograd.grad(P[:, i, j].sum(), F, retain_graph=True)[0]
                        for i in range(3) for j in range(3)]
                C = torch.stack(rows, 1).reshape(-1, 3, 3, 3, 3)
                h = torch.einsum('e,eiJkL,eaJ,ebL->eaibk', self.vol, C, self.grad, self.grad)
                H = torch.zeros((self.ndof, self.ndof), dtype=self.dtype, device=self.device)
                H.index_put_((self.rows, self.cols), h.reshape(-1), accumulate=True)
            local = torch.einsum('e,eiJ,eaJ->eai', self.vol, P, self.grad)
            r = torch.zeros_like(x).index_add(0, self.elements.flatten(), local.reshape(-1, 3))
            return (self.vol*W).sum(), r, H

    def old_bulk(self, x, theta, tangent=False):
        material = torch.exp(self.feature @ theta[self.i_material:self.i_field])
        mu = material/2.7; lam = material*.35/(1.35*.3)
        F = torch.einsum('eai,eaj->eij', x[self.elements], self.grad)
        logJ = torch.log(torch.linalg.det(F)); invT = torch.linalg.inv(F).transpose(-1, -2)
        W = .5*mu*((F*F).sum((-1,-2))-3)-mu*logJ+.5*lam*logJ**2
        P = mu[:,None,None]*F+(lam*logJ-mu)[:,None,None]*invT
        local = torch.einsum('e,eiJ,eaJ->eai', self.vol, P, self.grad)
        r = torch.zeros_like(x).index_add(0, self.elements.flatten(), local.reshape(-1,3))
        H = None
        if tangent:
            D = torch.einsum('eiJ,eaJ->eai', invT, self.grad)
            h = torch.einsum('e,eaJ,ebJ,ik->eaibk', self.vol*mu, self.grad, self.grad, self.eye)
            h += torch.einsum('e,eak,ebi->eaibk', self.vol*(mu-lam*logJ), D, D)
            h += torch.einsum('e,eai,ebk->eaibk', self.vol*lam, D, D)
            H = torch.zeros((self.ndof,self.ndof), dtype=self.dtype, device=self.device)
            H.index_put_((self.rows,self.cols), h.reshape(-1), accumulate=True)
        return (self.vol*W).sum(), r, H

    def evaluate(self, x, theta, frame, tangent=False):
        E, r, H = super().evaluate(x, theta, frame, tangent)
        oldE, oldr, oldH = self.old_bulk(x, theta, tangent)
        newE, newr, newH = self.bulk(x, theta, tangent)
        return E-oldE+newE, r-oldr+newr, None if not tangent else H-oldH+newH


def parameters(model, learned=True):
    initial = np.array([-4.] + [2.]*len(model.weights) + [5.] + [0.]*model.n_material + [0.])
    bounds = [(-9.,1.)]+[(-3.,8.)]*len(model.weights)+[(0.,9.)]+[(-2.,2.)]*model.n_material+[(0.,0.)]
    if learned:
        initial = np.r_[initial, 0., -1., 0.]
        bounds += [(-2.,2.), (-1.,1.), (-1.5,1.5)]
    return initial, bounds
