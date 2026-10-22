"""Physical-time BDF2/BE with the original full consistent-mass P2 equations."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class P2State:
    u:torch.Tensor
    v:torch.Tensor
    time_s:float=0.


def step_p2(model,current,previous,h,force,method='BDF2',force_tolerance=1e-10,physical_force_tolerance=1e-9):
    if method not in ['BE','BDF2'] or not np.isfinite(h) or h<=0:raise ValueError('Declaredmethod and positive finite timestep required')
    if method=='BDF2' and previous is not None:
        heff=2*h/3;delta=current.u-previous.u
        target=current.u+delta/3+heff*(4*current.v-previous.v)/3
    else:heff=h;target=current.u+h*current.v
    answer=model.solve(force=force,initial=current.u,target=target,effective_dt=heff,force_tolerance=force_tolerance)
    u=answer['displacement'];increment=u-current.u
    if method=='BDF2' and previous is not None:
        v=(3*increment-(current.u-previous.u))/(2*h)
        acceleration=(3*v-4*current.v+previous.v)/(2*h)
    else:v=increment/h;acceleration=(v-current.v)/h
    residual=model.mass(acceleration)+answer['cache']['gradient']-force
    R=float(torch.linalg.vector_norm(residual[~model.mask]));clamp=float(abs(u[model.mask]).max()) if bool(model.mask.any()) else 0.
    if not bool(torch.isfinite(v).all()) or R>physical_force_tolerance or clamp!=0.:raise RuntimeError('Original physical transient residual/constraint failed: '+str((R,clamp)))
    kinetic=.5*float((v*model.mass(v)).sum())
    row=dict(time_s=current.time_s+h,h_s=h,effective_dt_s=heff,method='BE_STARTUP' if method=='BDF2' and previous is None else method,
             original_physical_free_R_N=R,original_objective_free_R_N=float(torch.linalg.vector_norm(answer['residual'])),fixed_displacement_error_m=clamp,
             minimum_J=float(answer['cache']['minimum_J']),internal_potential_energy_J=float(answer['cache']['energy']),elastic_energy_J=float(answer['cache']['energy']-answer['cache'].get('contact_energy',0.)),contact_energy_J=float(answer['cache'].get('contact_energy',0.)),kinetic_energy_J=kinetic,
             Newton_iterations=answer['iterations'],CG_iterations=sum(x.get('cg_iterations',0) for x in answer['history']))
    return P2State(u,v,current.time_s+h),row
