"""Isolated audited object backend: explicit floor hypothesis and fixed solve path.

No changes to the running v01 object bank. Floor-off removes its unobservable
floor latent entirely. All returned frames follow the same canonical continuation
through every command frame from zero, regardless of requested output sampling.
"""
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend


class ObjectBoundaryBackendRevision(ObjectBoundaryBackend):
    def __init__(self,payload,device='cpu',*,floor_mode='off'):
        if floor_mode not in ('off','inferred'):
            raise ValueError('Explicit floor_mode must be off or inferred')
        super().__init__(payload,device)
        self.floor_mode=floor_mode
        self._full_lower=self.lower.copy();self._full_upper=self.upper.copy()
        self._indices=np.arange(self.parameter_dim)
        if floor_mode=='off':self._indices=np.delete(self._indices,self.model.i_floor)
        self.names=[self.names[i]for i in self._indices]
        self.lower=self.lower[self._indices];self.upper=self.upper[self._indices]
        self.parameter_dim=len(self._indices)
        self.last_path_diagnostics=None

    def bind(self,logits):
        z=self.model.t(logits)
        if z.shape!=(self.parameter_dim,) or not bool(torch.isfinite(z).all()):
            raise ValueError('Finite latent vector required')
        full_z=torch.zeros(len(self._full_lower),dtype=self.model.dtype,device=self.model.device)
        full_z[torch.as_tensor(self._indices,device=self.model.device)]=z
        p=self.model.t(self._full_lower)+(self.model.t(self._full_upper)-self.model.t(self._full_lower))*torch.sigmoid(full_z)
        widths=torch.exp(p[2+self.groups:])
        raw=self.base_weights.clamp_min(1e-30).pow(widths[:,None])
        self.model.weights=raw/raw.sum(-1,keepdim=True)
        theta=torch.zeros(self.model.i_field+1,dtype=self.model.dtype,device=self.model.device)
        theta[:2+self.groups]=p[:2+self.groups]
        if self.floor_mode=='off':theta[self.model.i_floor]=-torch.inf
        return theta

    @torch.no_grad()
    def rollout(self,logits,frames,maxiter=60):
        frames=np.asarray(frames)
        if frames.ndim!=1 or len(frames)==0 or frames.dtype.kind not in 'iu':
            raise ValueError('Nonempty integer command-frame array required')
        if frames[0]<0 or frames[-1]>=len(self.data['controller_targets']) or np.any(np.diff(frames)<=0):
            raise ValueError('Strictly increasing in-range command frames required')
        theta=self.bind(logits);state=None;positions=[];diagnostics=[]
        requested=set(frames.tolist());all_diagnostics=[]
        for frame in range(int(frames[-1])+1):
            state,_,diag=self.model.solve(theta,frame,state,maxiter=maxiter)
            if not np.isfinite(diag['residual']) or diag['residual']>1e-5:
                raise RuntimeError('Unqualified nonlinear solve')
            all_diagnostics.append(diag)
            if frame in requested:
                positions.append(state.detach().cpu().numpy()*self.model.scale+self.model.origin)
                diagnostics.append(diag)
        self.last_path_diagnostics={'frames_solved':int(frames[-1])+1,'maximum_residual':max(d['residual']for d in all_diagnostics),'minimum_J':min(d['minJ']for d in all_diagnostics),'iterations':sum(d['iterations']for d in all_diagnostics)}
        return np.stack(positions),diagnostics
