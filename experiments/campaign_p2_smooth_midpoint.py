"""NEW implicit midpoint mechanics with explicit actuator work accounting.

M(v1-v0)/h + grad U((u0+u1)/2) = F(tmid),
u1-u0 = h(v0+v1)/2.  Solve at umid with heff=h/2 and
inertial target u0+(h/2)v0. No physical damping or mode projection.
For nonlinear U, midpoint is NOT exactly energy-conserving: the ledger retains
U1-U0-gradU_mid·(u1-u0) separately from stationarity work.
"""
import numpy as np
import torch
from experiments.campaign_p2_transient import P2State


def smooth_bump(t,duration_s=1.):
    """Public C3 compact pulse, unit peak; t/material/support independent."""
    if not np.isfinite([t,duration_s]).all() or duration_s<=0:raise ValueError('Finite public time and positive duration')
    s=float(t)/duration_s
    if s<=0 or s>=1:return 0.
    return 256*s**4*(1-s)**4


def midpoint_step(model,current,h,force_mid,*,force_tolerance_N=1e-10,physical_tolerance_N=1e-9):
    if not np.isfinite(h) or h<=0:raise ValueError('Positive finite h required')
    u0,v0=current.u,current.v
    if any(a.shape!=model.rest.shape or a.dtype!=torch.float64 or a.device!=model.rest.device or not bool(torch.isfinite(a).all()) for a in [u0,v0,force_mid]):raise ValueError('Finite matching FP64 nodal states/explicit force required')
    if bool((u0[model.mask]!=0).any()) or bool((v0[model.mask]!=0).any()):raise ValueError('Exact fixed initial u/v required')
    before=model.evaluate(u0);heff=h/2;target=u0+heff*v0
    answer=model.solve(force=force_mid,initial=u0,target=target,effective_dt=heff,force_tolerance=force_tolerance_N)
    mid=answer['displacement'];delta=mid-u0;u1=u0+2*delta;v1=4*delta/h-v0;endpoint=model.evaluate(u1)
    a=(v1-v0)/h;Ma=model.mass(a);res=Ma+answer['cache']['gradient']-force_mid
    R=float(torch.linalg.vector_norm(res[~model.mask]));objective_R=float(torch.linalg.vector_norm(answer['residual']));u_error=float(abs(u1[model.mask]).max()) if bool(model.mask.any()) else 0.;v_error=float(abs(v1[model.mask]).max()) if bool(model.mask.any()) else 0.
    if min(float(answer['cache']['minimum_J']),float(endpoint['minimum_J']))<=model.minimum_J:raise RuntimeError('Midpoint/endpoint positive-J guard failed')
    if not bool(torch.isfinite(u1).all()) or not bool(torch.isfinite(v1).all()) or R>physical_tolerance_N or objective_R>force_tolerance_N or u_error!=0 or v_error!=0:raise RuntimeError('Original midpoint residual/constraint/finite gate failed')
    # Full M, including M_cf, supplies physical fixed reactions.
    du=2*delta;K0=.5*(v0*model.mass(v0)).sum();K1=.5*(v1*model.mass(v1)).sum();dU=endpoint['energy']-before['energy'];work=(force_mid*du).sum();elastic_midwork=(answer['cache']['gradient']*du).sum();defect=dU-elastic_midwork;residual_work=(res*du).sum();balance=dU+K1-K0-work
    row=dict(time_s=current.time_s+h,midpoint_time_s=current.time_s+h/2,h_s=h,effective_dt_s=heff,original_physical_free_R_N=R,original_objective_free_R_N=objective_R,fixed_displacement_error_m=u_error,fixed_velocity_error_m_s=v_error,minimum_midpoint_J=float(answer['cache']['minimum_J']),minimum_endpoint_J=float(endpoint['minimum_J']),elastic_energy_J=float(endpoint['energy']),kinetic_energy_J=float(K1),external_midpoint_work_J=float(work),nonlinear_midpoint_internal_energy_defect_J=float(defect),stationarity_work_J=float(residual_work),total_energy_minus_midpoint_work_J=float(balance),ledger_identity_defect_J=float(balance-defect-residual_work),fixed_reaction_resultant_N=res.masked_fill(~model.mask,0.).sum(0).detach().cpu().numpy().tolist(),Newton_iterations=answer['iterations'],CG_iterations=sum(x.get('cg_iterations',0) for x in answer['history']))
    return P2State(u1,v1,current.time_s+h),row
