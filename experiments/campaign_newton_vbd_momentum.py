"""Float64 host momentum recovery from supplied literal forces; no q projection.

For an exact BE/BDF2 solve this equals the position derivative. With finite q
precision it avoids differentiating quantization noise. Its momentum residual is
by construction; kinematic defect and independent forward comparisons remain
necessary. No solver, observations, truth or fitted parameter access here.
"""
import numpy as np


def momentum_velocity(method,step,h,v,oldv,internal,external,mass,active):
    v=np.asarray(v,dtype=np.float64);f=np.asarray(internal,dtype=np.float64)+np.asarray(external,dtype=np.float64)
    m=np.asarray(mass,dtype=np.float64);a=np.asarray(active)
    if method not in ['BE','BDF2'] or type(step) is not int or step<1 or not np.isfinite(h) or h<=0 or v.ndim!=2 or v.shape[1]!=3 or f.shape!=v.shape or m.shape!=(len(v),) or a.shape!=(len(v),) or a.dtype!=np.bool_ or not all(np.isfinite(x).all() for x in [v,f,m]) or np.any(m<=0):raise ValueError('Explicit finite state/force, positive mass and active-mask required')
    if method=='BE' or step==1:answer=v+h*f/m[:,None]
    else:
        previous=np.asarray(oldv,dtype=np.float64)
        if previous.shape!=v.shape or not np.isfinite(previous).all():raise ValueError('Independent previous physical velocity required')
        answer=(4*v-previous)/3+(2*h/3)*f/m[:,None]
    answer[~a]=0.
    return answer


def kinematic_defect(method,step,h,qnew,q,oldq,vnew):
    new=np.asarray(qnew,dtype=np.float64);current=np.asarray(q,dtype=np.float64);velocity=np.asarray(vnew,dtype=np.float64)
    if method not in ['BE','BDF2'] or step<1 or not np.isfinite(h) or h<=0 or new.shape!=current.shape or velocity.shape!=new.shape or not all(np.isfinite(a).all() for a in [new,current,velocity]):raise ValueError('Explicit aligned finite kinematic states required')
    if method=='BE' or step==1:return (new-current)-h*velocity
    previous=np.asarray(oldq,dtype=np.float64)
    if previous.shape!=new.shape or not np.isfinite(previous).all():raise ValueError('Independent previous position required')
    # Algebraically qnew - ((4q-qprev)/3 + heff*vnew), in stable increments.
    return (new-current)-(current-previous)/3-(2*h/3)*velocity


def kinematic_position(method,step,h,previous,oldprevious,vnew):
    """Integrate recovered velocity separately; never overwrite native positions."""
    q=np.asarray(previous,dtype=np.float64);v=np.asarray(vnew,dtype=np.float64)
    if method not in ['BE','BDF2'] or step<1 or not np.isfinite(h) or h<=0 or q.shape!=v.shape or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Explicit finite kinematic state required')
    if method=='BE' or step==1:return q+h*v
    old=np.asarray(oldprevious,dtype=np.float64)
    if old.shape!=q.shape or not np.isfinite(old).all():raise ValueError('Independent previous integrated position required')
    return q+(q-old)/3+(2*h/3)*v
