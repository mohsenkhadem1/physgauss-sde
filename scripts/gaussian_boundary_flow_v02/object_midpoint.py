"""Implicit midpoint mechanics; raw states, physical forces, no impact projection.

Solve V(y,t+h/2) + 2/h² ||y-x-h*v/2||²_M for midpoint y.
Then x_new=2*y-x and v_new=2*(x_new-x)/h-v. This is second order
on smooth forces, not an unconditional accuracy guarantee across contact onset.
"""
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend


class MidpointObjectBoundaryBackend(DynamicObjectBoundaryBackend):
    @torch.no_grad()
    def rollout(self,logits,frames,maxiter=60,initial_velocity=None):
        frames=np.asarray(frames)
        if frames.ndim!=1 or not len(frames) or frames.dtype.kind not in 'iu' or frames[0]<0 or frames[-1]>=len(self.times) or np.any(np.diff(frames)<=0):
            raise ValueError('Increasing valid integer frames required')
        m=self.model;theta=self.bind(logits);x=m.rest.clone()
        v=torch.zeros_like(x) if initial_velocity is None else m.t(initial_velocity)/m.scale
        if v.shape!=x.shape or not bool(torch.isfinite(v).all()):raise ValueError('Finite initial velocities required')
        requested=set(frames.tolist());positions=[];diagnostics=[];all_diag=[]
        energies=[];maximum_penetration=0.
        def energy(state,velocity):
            old_target,old_dt=m.inertial_target,m.inertial_dt
            m.inertial_target=state;m.inertial_dt=1.
            potential,_,_=m.evaluate(state,theta,0)
            m.inertial_target=old_target;m.inertial_dt=old_dt
            kinetic=.5*torch.exp(theta[0])*m.scale/9.81*(m.mass[:,None]*velocity**2).sum()
            return float(potential+kinetic)
        energies.append(energy(x,v))
        if 0 in requested:
            positions.append(x.cpu().numpy()*m.scale+m.origin)
            diagnostics.append({'iterations':0,'residual':None,'minJ':m.min_jacobian(x),'initial_state':True})
        try:
            for frame in range(1,int(frames[-1])+1):
                dt=(self.times[frame]-self.times[frame-1])/self.substeps
                for sub in range(self.substeps):
                    fraction=(sub+.5)/self.substeps
                    m.targets=((1-fraction)*self.command_targets[frame-1]+fraction*self.command_targets[frame])[None]
                    # Activation jumps at command timestamps; midpoint lies before right endpoint.
                    m.active=self.command_active[frame-1][None]
                    m.inertial_target=x+.5*dt*v;m.inertial_dt=.5*dt
                    mid,_,diag=m.solve(theta,0,x,tolerance=1e-10,maxiter=maxiter)
                    if not np.isfinite(diag['residual']) or diag['residual']>1e-7:raise RuntimeError(f"Midpoint residual tolerance not met: {diag['residual']:.9g} at frame {frame}, substep {sub}")
                    nxt=2*mid-x;endpoint_J=m.min_jacobian(nxt)
                    if not np.isfinite(endpoint_J) or endpoint_J<=.03:raise RuntimeError('Invalid midpoint endpoint')
                    v=2*(nxt-x)/dt-v;x=nxt
                    energies.append(energy(x,v))
                    maximum_penetration=max(maximum_penetration,float((m.floor-x[:,2]).clamp_min(0).max())*m.scale)
                    diag={**diag,'midpoint_minJ':diag['minJ'],'minJ':endpoint_J}
                    all_diag.append(diag)
                if frame in requested:
                    positions.append(x.cpu().numpy()*m.scale+m.origin);diagnostics.append(diag)
            self.last_velocity=v.cpu().numpy()*m.scale
            self.last_energies=np.asarray(energies)
            self.last_maximum_penetration=maximum_penetration
            self.last_path_diagnostics={'integrator':'implicit_midpoint','substeps':self.substeps,'steps':len(all_diag),'minimum_J':min([min(d['minJ'],d['midpoint_minJ']) for d in all_diag],default=1.),'max_residual':max([d['residual']for d in all_diag],default=0.),'raw_mechanical_states':True,'rebasing':False}
            return np.stack(positions),diagnostics
        finally:
            m.targets=self.command_targets;m.active=self.command_active;m.inertial_target=None;m.inertial_dt=None
