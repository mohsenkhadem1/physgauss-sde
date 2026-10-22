"""NEW small nonlinear homogeneous-bar inverse; not a Gaussian/4D rescue.

Known support location/law, unknown E/K. Raw left/right observations remain
independent; extension differences are NOT counted as extra observations.
"""
from __future__ import annotations
import numpy as np
from scipy.special import lambertw
from experiments.campaign_static_uniaxial_bar import SIZE,POISSON,analytic_solution

LOWER=np.log([4500.,100.]);UPPER=np.log([9000.,400.]);TRUTH=np.array([6000.,220.]);SIGMA=.0002
FORCES=(1.,3.);HELD_FORCE=2.


def physical(unit):
    x=np.asarray(unit,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or ((x<0)|(x>1)).any():raise ValueError('Finite prior-unit[E,K] coordinates required')
    return np.exp(LOWER+x*(UPPER-LOWER))


def stretch(young,forces):
    """Exact finite-NH uniaxial law, converged float64 vector Newton roots.

    Lateral y=c*W(exp(1/c)/(c*s)) is the positive real branch. Newton targets
    the ORIGINAL force equation; every candidate must meet the force tolerance.
    """
    young=np.asarray(young,dtype=float);forces=np.asarray(forces,dtype=float)
    if young.ndim!=1 or not len(young) or forces.ndim!=1 or not len(forces) or not np.isfinite(young).all() or not np.isfinite(forces).all() or (young<=0).any() or (forces<0).any():raise ValueError('PositiveE and nonnegative known forces required')
    area=SIZE[1]*SIZE[2];mu=young[:,None]/(2*(1+POISSON));c=2*POISSON/(1-2*POISSON)
    s=1+forces[None]/(young[:,None]*area)
    for _ in range(12):
        y=c*lambertw(np.exp(1/c)/(c*s)).real
        residual=mu*(s-y/s)-forces[None]/area
        if np.max(abs(residual*area))<=1e-12:return s,y
        derivative=mu*(1+y/s**2*(1+c/(c+y)))
        s=s-residual/derivative
        if np.any(s<1-1e-13) or np.any(s>1.5) or not np.isfinite(s).all():raise ValueError('Original force-root escaped declared stable bracket')
    raise RuntimeError('Original nonlinear force residual did not converge')


def predict(theta,forces=FORCES):
    theta=np.asarray(theta,dtype=float);forces=np.asarray(forces,dtype=float)
    if theta.ndim!=2 or theta.shape[1]!=2 or not len(theta) or not np.isfinite(theta).all() or (theta<=0).any():raise ValueError('Positive finite physical E/K required')
    s,_=stretch(theta[:,0],forces);left=forces[None]/theta[:,1,None];right=left+SIZE[0]*(s-1)
    return np.stack([left,right],axis=2).reshape(len(theta),-1)


def independent_predict(theta,forces=FORCES):
    rows=[]
    for young,spring in np.asarray(theta,dtype=float):
        out=[]
        for force in forces:
            answer=analytic_solution(force,young=young,spring=spring)
            out.extend([answer['left_translation_m'],answer['right_translation_m']])
        rows.append(out)
    return np.asarray(rows)


def log_likelihood(unit,observed):
    y=np.asarray(observed,dtype=float)
    if y.shape!=(4,) or not np.isfinite(y).all():raise ValueError('Four acquired left/right displacements required')
    error=(predict(physical(unit))-y)/SIGMA
    return -.5*np.sum(error**2,axis=1)-4*np.log(SIGMA*np.sqrt(2*np.pi))
