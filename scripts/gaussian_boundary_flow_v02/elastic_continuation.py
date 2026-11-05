"""Experimental arbitrary-state implicit-midpoint continuation of an existing FEM.

Uses the SAME object/model, reference mesh, parameters and potential. Removes the
legacy static numerical gauge during inertia through DynamicFEM, or explicitly for
an ExternalFEM instance. No contact handling: encountering the declared plane fails.
No rest resets, pose projections, shape freezing or parameter estimation.
"""
import numpy as np
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.external_benchmark.fem import ExternalFEM


def continue_elastic(model,theta,positions_m,velocities_m_s,start_time_s,output_times_s,
                     *,max_dt,command_times_s=None,targets=None,activation=None,
                     reject_floor=True,maxiter=80):
    if not isinstance(model,ExternalFEM):raise ValueError('Existing ExternalFEM-compatible mechanics instance required')
    if not reject_floor:raise ValueError('Contact safety cannot be disabled on this unqualified continuation')
    theta=model.t(theta)
    if not bool(torch.isfinite(theta[0])) or not bool(torch.isfinite(torch.exp(theta[0]))) or float(torch.exp(theta[0]))<=0:raise ValueError('Positive finite inertial mass scale required')
    if isinstance(maxiter,bool)or not isinstance(maxiter,int)or maxiter<1:raise ValueError('Positive integer solve budget required')
    times=np.asarray(output_times_s,float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[0]<start_time_s or not np.isfinite(start_time_s) or not np.isfinite(max_dt) or max_dt<=0:
        raise ValueError('Finite increasing output times, finite start and positive max_dt required')
    x=model.t((np.asarray(positions_m)-model.origin)/model.scale).clone()
    v=model.t(velocities_m_s).clone()/model.scale
    if x.shape!=model.rest.shape or v.shape!=x.shape or not torch.isfinite(x).all() or not torch.isfinite(v).all():raise ValueError('Finite nodal positions/velocities required')
    if model.min_jacobian(x)<=.03:raise ValueError('Invalid supplied initial elastic state')
    if targets is None and not torch.equal(model.targets,model.targets[:1].expand_as(model.targets)):
        raise ValueError('Explicit prescribed targets required for varying model commands')
    if activation is None and not torch.equal(model.active,model.active[:1].expand_as(model.active)):
        raise ValueError('Explicit activation schedule required for varying model commands')
    if command_times_s is None and (targets is not None or activation is not None):
        raise ValueError('Explicit command timestamps required with supplied schedules')
    command_times=np.asarray([start_time_s,times[-1]+max_dt]if command_times_s is None else command_times_s,float)
    target=model.targets[:1].expand(len(command_times),-1,-1,-1).clone()if targets is None else model.t((np.asarray(targets)-model.origin)/model.scale)
    active=model.active[:1].expand(len(command_times),-1).clone()if activation is None else model.t(activation)
    if command_times.ndim!=1 or len(command_times)<2 or not np.isfinite(command_times).all() or np.any(np.diff(command_times)<=0)or command_times[0]>start_time_s or command_times[-1]<times[-1]:raise ValueError('Commands must cover time interval')
    if target.shape!=(len(command_times),len(model.weights),model.n,3)or active.shape!=(len(command_times),len(model.weights))or not torch.isfinite(target).all()or not torch.isfinite(active).all()or (active<0).any()or (active>1).any():raise ValueError('Invalid prescribed command arrays')
    # A fixed step grid depends on start/max_dt and command events, never requested outputs.
    stop=float(times[-1]);events=sorted(set([float(start_time_s),stop]+[float(t)for t in command_times if start_time_s<t<stop]))
    grid=[float(start_time_s)]
    for a,b in zip(events[:-1],events[1:]):
        count=max(1,int(np.ceil((b-a)/max_dt-1e-12)));grid.extend(np.linspace(a,b,count+1)[1:].tolist())
    indices=[]
    for t in times:
        i=int(np.argmin(abs(np.asarray(grid)-t)))
        if abs(grid[i]-t)>1e-10:raise ValueError('Output times must lie on fixed integration grid; no interpolation or output-dependent steps')
        indices.append(i)
    original=(model.targets,model.active,model.inertial_target,model.inertial_dt)
    saved=[];velocities=[];diagnostics=[];energies=[]
    mass=torch.exp(theta[0])*model.scale/9.81*model.mass
    def set_command(t):
        i=min(len(command_times)-2,max(0,int(np.searchsorted(command_times,t,side='right')-1)))
        f=(t-command_times[i])/(command_times[i+1]-command_times[i])
        model.targets=((1-f)*target[i]+f*target[i+1])[None];model.active=active[i][None]
    def evaluate(q,tangent=False):
        E,r,H=model.evaluate(q,theta,0,tangent)
        if not isinstance(model,DynamicFEM):
            delta=q-model.rest;E=E-.5e-5*delta.square().sum();r=r-1e-5*delta
            if H is not None:H=H-1e-5*torch.eye(model.ndof,dtype=model.dtype,device=model.device)
        return E,r,H
    def energy(q,w):
        old=model.inertial_target;model.inertial_target=q
        E,_,_=evaluate(q);model.inertial_target=old
        return float(E+.5*(mass[:,None]*w.square()).sum())
    def contact(q):
        if reject_floor and bool((q[:,2]<=model.floor+1e-10).any()):raise RuntimeError('Unsupported plane contact encountered; no projection or impact response')
    try:
        model.inertial_dt=1.;model.inertial_target=x;set_command(start_time_s);contact(x)
        energies.append(energy(x,v));states=[x.clone()];speeds=[v.clone()]
        with torch.no_grad():
            for a,b in zip(grid[:-1],grid[1:]):
                h=b-a;set_command((a+b)/2);model.inertial_target=x+.5*h*v;model.inertial_dt=.5*h
                y=x.clone();residual=float('inf')
                for iteration in range(maxiter):
                    E,r,H=evaluate(y,True);residual=float(torch.linalg.vector_norm(r))
                    if not np.isfinite(residual):raise RuntimeError('Nonfinite midpoint residual')
                    correction=None
                    try:
                        probe=torch.linalg.solve(H,-r.flatten())
                        if bool(torch.isfinite(probe).all()):correction=float(probe.abs().max())
                    except RuntimeError:pass
                    if residual<=1e-8 or (residual<=1e-7 and correction is not None and correction<=1e-9):break
                    accepted=False;scale=torch.diag(H).abs().mean().clamp_min(1e-6)
                    for damping in (0.,1e-8,1e-5,1e-3,.1,1.,10.):
                        try:d=torch.linalg.solve(H+damping*scale*torch.eye(model.ndof,dtype=model.dtype,device=model.device),-r.flatten()).reshape_as(y)
                        except RuntimeError:continue
                        slope=float((r*d).sum())
                        if not torch.isfinite(d).all()or slope>=0:continue
                        for ls in range(24):
                            alpha=2.**(-ls)
                            try:En,rn,_=evaluate(y+alpha*d)
                            except(ValueError,RuntimeError):continue
                            small=(residual<=1e-7 or (correction is not None and correction<=1e-9)) and float(torch.linalg.vector_norm(rn))<.5*residual and float(En)<=float(E)+64*torch.finfo(model.dtype).eps*max(1.,abs(float(E)))
                            if float(En)<=float(E)+1e-4*alpha*slope or small:y=y+alpha*d;accepted=True;break
                        if accepted:break
                    if not accepted:raise RuntimeError(f'Midpoint solve failed residual={residual}')
                else:raise RuntimeError(f'Midpoint iteration limit residual={residual}, correction={correction}')
                nxt=2*y-x;contact(y);contact(nxt)
                J=model.min_jacobian(nxt)
                if not np.isfinite(J)or J<=.03:raise RuntimeError('Invalid elastic endpoint')
                v=2*(nxt-x)/h-v;x=nxt
                states.append(x.clone());speeds.append(v.clone());energies.append(energy(x,v));diagnostics.append(dict(time_s=b,residual=residual,minJ=J))
        for i in indices:
            saved.append(states[i].cpu().numpy()*model.scale+model.origin);velocities.append(speeds[i].cpu().numpy()*model.scale)
        return dict(positions_m=np.stack(saved),velocities_m_s=np.stack(velocities),times_s=times,step_times_s=np.asarray(grid),energies=np.asarray(energies),steps=diagnostics,scope='Experimental same-model no-impact elastic continuation; energies include active boundary potential; driving/switch work not yet separated')
    finally:model.targets,model.active,model.inertial_target,model.inertial_dt=original
