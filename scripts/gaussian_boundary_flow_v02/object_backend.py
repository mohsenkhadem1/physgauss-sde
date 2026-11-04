"""Geometry-specific nonlinear boundary-inference backend for exposed real objects.

Reuses the existing finite-strain FEM. Latents are dimensionless response ratios:
video without measured mass/forces cannot identify an absolute material scale.
This is quasistatic and does NOT replace the validated dynamic block experiment.
"""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM


class ObjectBoundaryBackend:
    def __init__(self,payload,device='cpu'):
        if 'evaluation' in payload:raise ValueError('Evaluation targets forbidden')
        if len(payload['observed'])!=payload['train_end']:raise ValueError('Prefix-only observations required')
        self.data=payload
        self.model=ExternalFEM(payload,device=device,field=False,contact=False,nonconvex=False,materials=1)
        self.groups=len(self.model.weights)
        self.base_weights=self.model.weights.clone()
        # Explicit finite-strain convex-hull pilot; nonconvex mesh is a later gate.
        self.names=['log_gravity_material_ratio']+[f'log_attachment_material_ratio_{i}' for i in range(self.groups)]+['log_floor_material_ratio']+[f'log_inverse_width_squared_{i}'for i in range(self.groups)]
        self.lower=np.array([-8.]+[0.]*self.groups+[3.]+[-.8]*self.groups)
        self.upper=np.array([-1.]+[7.]*self.groups+[8.]+[.8]*self.groups)
        self.parameter_dim=len(self.lower)

    def bind(self,logits):
        z=self.model.t(logits)
        if z.shape!=(self.parameter_dim,) or not bool(torch.isfinite(z).all()):raise ValueError('Finite latent vector required')
        p=self.model.t(self.lower)+(self.model.t(self.upper)-self.model.t(self.lower))*torch.sigmoid(z)
        widths=torch.exp(p[2+self.groups:])
        # Raising a Gaussian to a positive power changes covariance, preserving
        # its center. Normalize to separate footprint from overall stiffness.
        raw=self.base_weights.clamp_min(1e-30).pow(widths[:,None])
        self.model.weights=raw/raw.sum(-1,keepdim=True)
        theta=torch.zeros(self.model.i_field+1,dtype=self.model.dtype,device=self.model.device)
        theta[:2+self.groups]=p[:2+self.groups]
        return theta

    @torch.no_grad()
    def rollout(self,logits,frames,maxiter=60):
        theta=self.bind(logits);state=None;positions=[];diagnostics=[]
        for frame in frames:
            state,_,diag=self.model.solve(theta,int(frame),state,maxiter=maxiter)
            if not np.isfinite(diag['residual']) or diag['residual']>1e-5:raise RuntimeError('Unqualified nonlinear solve')
            positions.append(state.detach().cpu().numpy()*self.model.scale+self.model.origin)
            diagnostics.append(diag)
        return np.stack(positions),diagnostics

    def observation_features(self,positions,frames,ids):
        # Interpolation is fixed from initial geometry only; no shape labels here.
        A=self.data['observation_matrix'][ids]
        displacement=positions-self.data['nodes'][None]
        return self.data['observed_initial'][ids][None]+np.einsum('on,tnc->toc',A,displacement)
