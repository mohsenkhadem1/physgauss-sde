"""Elastic release into actual initial static-scene heightfields.

A diagnostic alternative to frozen-shape pose dynamics. Neo-Hookean internal
energy and fitted parameters are retained. Contact remains vertical, penalized,
and lagged: no sidewalls, CCD/IPC, or guarantee of global nonintersection.
"""
import numpy as np
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.external_benchmark.data import predictor_payload


class SceneFEM(DynamicFEM):
    def __init__(self,data,scene,device='cpu',field=True,nonconvex=False,materials=1,
                 friction=.3,friction_smoothing_m=.0002,sweep_tolerance_m=.002,
                 disable_field_on_release=False):
        # contact=False disables the old reversible rest-anchored friction
        # hypothesis. Extra theta entries remain accepted and unchanged.
        super().__init__(data,device=device,field=field,contact=False,
                         nonconvex=nonconvex,materials=materials)
        self.active=self.active.clone()
        self.scene=scene;self.friction=float(friction)
        self.disable_field_on_release=bool(disable_field_on_release)
        self.friction_epsilon=float(friction_smoothing_m)/self.scale
        self.sweep_tolerance=float(sweep_tolerance_m)
        self.lagged_position=None;self.lagged_normal=None
        self.support_height=None

    def prepare_step(self,previous,theta,dt):
        """Freeze support query and friction normal force for one implicit step."""
        world=previous.detach().cpu().numpy()*self.scale+self.origin
        limit=np.maximum(world[:,2]+self.sweep_tolerance,self.scene.floor)
        if self.support_height is not None:
            # Keep a previously reached support eligible through small penalty
            # penetration; querying the new footprint still permits separation.
            limit=np.maximum(limit,self.support_height)
        query=self.scene.query_surface(world,max_height=limit)
        height=query['height'];height=np.where(np.isfinite(height),height,self.scene.floor)
        self.support_height=height.copy()
        self.floor=self.t((height-self.origin[2])/self.scale)
        self.lagged_position=previous.detach().clone()
        normal=torch.exp(theta[self.i_floor])*self.mass*torch.clamp(self.floor-previous[:,2],min=0.)
        self.lagged_normal=normal.detach()
        self.inertial_dt=float(dt)
        return query

    def evaluate(self,x,theta,frame,tangent=False):
        saved_field=self.field
        if self.disable_field_on_release and self.inertial_target is not None:self.field=False
        try:energy,residual,H=super().evaluate(x,theta,frame,tangent)
        finally:self.field=saved_field
        if self.lagged_position is None or self.friction==0:return energy,residual,H
        delta=x[:,:2]-self.lagged_position[:,:2]
        norm=torch.sqrt((delta*delta).sum(1)+self.friction_epsilon**2)
        strength=self.friction*self.lagged_normal
        energy=energy+(strength*(norm-self.friction_epsilon)).sum()
        grad=strength[:,None]*delta/norm[:,None]
        residual=residual+torch.cat([grad,torch.zeros((self.n,1),device=self.device,dtype=self.dtype)],1)
        if H is not None:
            eye=torch.eye(2,device=self.device,dtype=self.dtype)
            blocks=strength[:,None,None]*(eye[None]/norm[:,None,None]-delta[:,:,None]*delta[:,None,:]/norm[:,None,None]**3)
            ids=torch.arange(self.ndof,device=self.device).reshape(-1,3)[:,:2]
            H=H.index_put((ids[:,:,None].expand_as(blocks).reshape(-1),ids[:,None,:].expand_as(blocks).reshape(-1)),blocks.reshape(-1),accumulate=True)
        return energy,residual,H


def rollout(data,base_prediction,theta,scene,start=None,device='cpu',field=True,
            nonconvex=False,materials=1,friction=.3,substeps=(5,10,20,40),
            initial_state=None,initial_velocity=None,free_mask=None,disable_field_on_release=False):
    """Preserve held predictions, integrate elastic free motion after start.

    `initial_state` is world mesh-node geometry at start-1, reconstructed only
    from the allowed prefix by caller; optional velocity is metres/second.
    Base failures replaced by successful free dynamics remain in metadata.
    A failed frame remains NaN; continuation is stopped rather than teleporting
    through an unintegrated interval. This routine never reads evaluation.
    """
    base=np.asarray(base_prediction,float);pred=base.copy();end=len(base)
    start=int(data['train_end'] if start is None else start)
    if not 1<=start<end:raise ValueError('Need at least one initialization and future frame')
    model=SceneFEM(predictor_payload(data),scene,device,field,nonconvex,materials,friction,
                   disable_field_on_release=disable_field_on_release)
    theta=model.t(theta)
    state=base[start-1] if initial_state is None else np.asarray(initial_state)
    if not np.isfinite(state).all():raise ValueError('Finite prefix initialization required')
    x=model.t((state-model.origin)/model.scale)
    dt0=float(data['times'][start-1]-data['times'][start-2]) if start>1 else float(data['times'][1]-data['times'][0])
    if initial_velocity is None:
        velocity=(state-base[start-2])/dt0 if start>1 and np.isfinite(base[start-2]).all() else np.zeros_like(state)
    else:velocity=np.asarray(initial_velocity)
    velocity=model.t(velocity/model.scale)
    free=(np.asarray(data['attachment_active']).sum(1)==0) if free_mask is None else np.asarray(free_mask,bool)
    # Physical detachment is an explicitly caller-supplied input hypothesis.
    model.active[torch.as_tensor(free,device=device)]=0
    diagnostics=[];failed=[]
    for frame in range(start,end):
        if not free[frame]:
            if not np.isfinite(base[frame]).all():
                failed.append(dict(frame=frame,reason='Held base frame failed'));pred[frame:]=np.nan;break
            old=model.t((base[frame-1]-model.origin)/model.scale)
            x=model.t((base[frame]-model.origin)/model.scale)
            velocity=(x-old)/float(data['times'][frame]-data['times'][frame-1]);model.support_height=None
            continue
        old=x.clone();old_velocity=velocity.clone();old_height=None if model.support_height is None else model.support_height.copy()
        errors=[];accepted=False
        for steps in substeps:
            x=old.clone();velocity=old_velocity.clone();model.support_height=None if old_height is None else old_height.copy()
            dt=float(data['times'][frame]-data['times'][frame-1])/int(steps)
            try:
                for _ in range(int(steps)):
                    model.prepare_step(x,theta,dt)
                    model.inertial_target=x+dt*velocity
                    previous=x;x,_,diag=model.solve(theta,frame,x)
                    velocity=(x-previous)/dt
                pred[frame]=x.cpu().numpy()*model.scale+model.origin
                diagnostics.append(dict(frame=frame,substeps=int(steps),**diag,prior_attempts=errors,
                                        final_substep_contact_nodes=int((model.lagged_normal>0).sum()),
                                        final_substep_normal_sum=float(model.lagged_normal.sum())))
                accepted=True;break
            except (RuntimeError,ValueError) as error:errors.append(dict(substeps=int(steps),reason=str(error)))
        if not accepted:
            pred[frame:]=np.nan;failed.append(dict(frame=frame,attempts=errors));break
    meta=dict(model='elastic Neo-Hookean FEM with lagged scene heightfield contact',
              start=start,friction=friction,disable_field_on_release=bool(disable_field_on_release),
              substeps=list(substeps),diagnostics=diagnostics,failed=failed,
              failed_frames=np.flatnonzero(~np.isfinite(pred).all(axis=(1,2))).tolist(),
              inherited_base_failed_frames=np.flatnonzero(~np.isfinite(base).all(axis=(1,2))).tolist(),
              limitations=['vertical contact only','lagged penalty surfaces','no continuous collision guarantee','fixed scene, no two-way rigid motion'])
    return pred,meta
