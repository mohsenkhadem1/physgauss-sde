"""Dynamic finite-strain continuation of Gaussian attachment/plane operators.

Physical timestamps drive implicit Euler. The existing DynamicFEM removes the
static positional gauge, so unsupported bodies really fall. Initial geometry is
an explicit state; no per-node rebasing of exported mechanical positions occurs.
"""
import numpy as np
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.gaussian_boundary_flow_v02.object_backend_revision import ObjectBoundaryBackendRevision


class DynamicObjectBoundaryBackend(ObjectBoundaryBackendRevision):
    def __init__(self,payload,device='cpu',*,floor_mode='inferred',substeps=4):
        super().__init__(payload,device,floor_mode=floor_mode)
        if not isinstance(substeps,int) or substeps<1:raise ValueError('Positive integer substeps required')
        self.substeps=substeps
        self.model=DynamicFEM(payload,device=device,field=False,contact=False,nonconvex=False,materials=1)
        self.base_weights=self.model.weights.clone()
        self.command_targets=self.model.targets.clone();self.command_active=self.model.active.clone()
        self.times=np.asarray(payload['times'],float)
        if len(self.times)!=len(self.command_targets) or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):raise ValueError('Strictly increasing finite command timestamps required')

    @torch.no_grad()
    def rollout(self,logits,frames,maxiter=60,initial_velocity=None):
        frames=np.asarray(frames)
        if frames.ndim!=1 or not len(frames) or frames.dtype.kind not in 'iu' or frames[0]<0 or frames[-1]>=len(self.times) or np.any(np.diff(frames)<=0):raise ValueError('Increasing valid integer frames required')
        m=self.model;theta=self.bind(logits);x=m.rest.clone()
        v=torch.zeros_like(x) if initial_velocity is None else m.t(initial_velocity)/m.scale
        if v.shape!=x.shape or not bool(torch.isfinite(v).all()):raise ValueError('Finite initial velocities in metres/second required')
        requested=set(frames.tolist());positions=[];diagnostics=[];all_diag=[]
        if 0 in requested:
            positions.append(x.cpu().numpy()*m.scale+m.origin)
            diagnostics.append({'iterations':0,'residual':None,'minJ':m.min_jacobian(x),'initial_state':True})
        try:
            for frame in range(1,int(frames[-1])+1):
                dt=(self.times[frame]-self.times[frame-1])/self.substeps
                for sub in range(1,self.substeps+1):
                    fraction=sub/self.substeps
                    m.targets=((1-fraction)*self.command_targets[frame-1]+fraction*self.command_targets[frame])[None]
                    # Right-continuous known activation commands, no interpolated contact truth.
                    m.active=(self.command_active[frame-1] if sub<self.substeps else self.command_active[frame])[None]
                    m.inertial_target=x+dt*v;m.inertial_dt=dt
                    nxt,_,diag=m.solve(theta,0,x,tolerance=1e-9,maxiter=maxiter)
                    if diag['residual']>1e-7:raise RuntimeError('Dynamic residual tolerance not met')
                    v=(nxt-x)/dt;x=nxt;all_diag.append(diag)
                if frame in requested:
                    positions.append(x.cpu().numpy()*m.scale+m.origin);diagnostics.append(diag)
            self.last_path_diagnostics={'substeps':self.substeps,'steps':len(all_diag),'minimum_J':min([d['minJ']for d in all_diag],default=1.),'max_residual':max([d['residual']for d in all_diag],default=0.),'raw_mechanical_states':True,'rebasing':False}
            return np.stack(positions),diagnostics
        finally:
            m.targets=self.command_targets;m.active=self.command_active;m.inertial_target=None;m.inertial_dt=None


def falling_cube_payload(duration=.4,step=.01):
    a=np.array([-.05,.05]);nodes=np.stack(np.meshgrid(a,a,a+.25,indexing='ij'),-1).reshape(-1,3)
    times=np.arange(round(duration/step)+1)*step
    return {'nodes':nodes,'observed':nodes[None],'train_end':1,'observed_initial':nodes,'observed_mask':np.ones((1,8),bool),'observation_matrix':np.eye(8),'controller_targets':np.empty((len(times),0,8,3)),'attachment_weights':np.empty((0,8)),'attachment_active':np.empty((len(times),0)),'floor':0.,'times':times,'controls':np.empty((len(times),0,3))}
