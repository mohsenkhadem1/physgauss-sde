"""Independent fully solved BE/BDF2 reference, stableNH P1/lumped mass.

Physical time and velocity are explicit; no static pseudo-time, damping,
constitutive repair, or change to sealed Newton/CPU sources.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_newton_stable_material import StableP1Block

class ImplicitFailure(RuntimeError):
    def __init__(self,message,history):super().__init__(message);self.history=history


def solve_inertial(elastic,q_start,target,mass,force,mask,h_eff,tolerance=1e-10,maximum_iterations=20):
    q=np.asarray(q_start,dtype=float).copy();target=np.asarray(target,dtype=float);mass=np.asarray(mass,dtype=float);force=np.asarray(force,dtype=float);mask=np.asarray(mask)
    if q.ndim!=2 or q.shape[1]!=3 or target.shape!=q.shape or force.shape!=q.shape or mask.shape!=q.shape or mask.dtype!=np.bool_ or mass.shape!=(len(q),) or np.any(mass<=0) or not np.isfinite(h_eff) or h_eff<=0 or not all(np.isfinite(z).all() for z in [q,target,mass,force]):raise ValueError('Finite consistent positive-mass implicit inputs required')
    free=np.flatnonzero(~mask.ravel());metric=mass[:,None]/h_eff**2;diagonal=np.repeat(metric[:,0],3);history=[]
    def evaluate(x,tangent):
        E,g,H,J=elastic(x,tangent=tangent)
        if not np.isfinite(J).all() or np.any(J<=.1):raise ValueError('PositiveJacobian guard failed; no repair')
        r=g+metric*(x-target)-force;d=x-target
        potential=float(E+.5*np.sum(metric*d*d)-np.sum(force*(x-q_start)))
        return potential,r,None if H is None else (H+sparse.diags(diagonal)).tocsr(),J,float(E),g
    for it in range(maximum_iterations+1):
        V,r,H,J,E,g=evaluate(q,True);norm=float(np.linalg.norm(r.ravel()[free]));history.append({'iteration':it,'original_objective_free_R_N':norm,'minimum_J':float(np.min(J))})
        if norm<=tolerance:return q,{'iterations':it,'objective_free_R_N':norm,'minimum_J':float(np.min(J)),'history':history,'elastic_energy_J':E,'elastic_gradient_N':g}
        if it==maximum_iterations:raise ImplicitFailure('Original implicit residual cap',history)
        try:direction=splu(H[free][:,free].tocsc()).solve(-r.ravel()[free])
        except RuntimeError as e:raise ImplicitFailure('Unmodified implicit tangent factor failed',history) from e
        slope=float(r.ravel()[free]@direction)
        if not np.isfinite(direction).all() or slope>=0:raise ImplicitFailure('Unmodified Newton direction not descent',history)
        for k in range(30):
            alpha=2.**(-k);trial=q.copy();trial.ravel()[free]+=alpha*direction
            try:VV,rr,_,_,_,_=evaluate(trial,False)
            except ValueError:continue
            roundoff=128*np.finfo(float).eps*max(abs(V),abs(E),1e-12)
            if VV<=V+1e-4*alpha*slope or(abs(VV-V)<=roundoff and np.linalg.norm(rr.ravel()[free])<norm):q=trial;history[-1]['alpha']=alpha;break
        else:raise ImplicitFailure('PositiveJ original-objective Armijo exhausted',history)
    raise AssertionError('Unreachable')


@dataclass
class TransientState:
    q:np.ndarray
    v:np.ndarray


def advance(elastic,mass,force,mask,current,h,method='BE',previous=None):
    if method not in ('BE','BDF2'):raise ValueError('Declared BE or BDF2 required')
    if method=='BDF2' and previous is not None:
        heff=2*h/3;target=current.q+(current.q-previous.q)/3+(2*h/9)*(4*current.v-previous.v);actual='BDF2'
    else:heff=h;target=current.q+h*current.v;actual='BE_STARTUP' if method=='BDF2' else'BE'
    target[mask]=current.q[mask]
    q,d=solve_inertial(elastic,current.q,target,mass,force,mask,heff)
    if actual=='BDF2':v=(3*(q-current.q)-(current.q-previous.q))/(2*h);acc=(3*v-4*current.v+previous.v)/(2*h)
    else:v=(q-current.q)/h;acc=(v-current.v)/h
    v[mask]=0.;physical=np.asarray(mass)[:,None]*acc+d['elastic_gradient_N']-force
    d.update(actual_method=actual,h_eff_s=heff,physical_original_free_R_N=float(np.linalg.norm(physical[~mask])),constraint_defect_m=float(np.max(abs((q-current.q)[mask]))) if mask.any() else 0.)
    d.pop('elastic_gradient_N');return TransientState(q,v),d


class StableBlockTransient:
    def __init__(self,cells=(4,2,2)):
        self.model=StableP1Block(cells,young=5000.,poisson=.35,density=625.,gravity=0.,traction=1.)
        self.mass=self.model.node_mass.copy();self.force=self.model.load.copy();self.mask=self.model.mask.copy()
    def initial(self):return TransientState(self.model.rest.copy(),np.zeros_like(self.model.rest))
    def step(self,state,h,method='BE',previous=None):return advance(self.model.elastic,self.mass,self.force,self.mask,state,h,method,previous)
