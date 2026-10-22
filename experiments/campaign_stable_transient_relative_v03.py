"""Numeric-only relative F and finite-energy-roundoff safeguard, same equations.

Old failed v01/v02 stay frozen. Same exact stable_material, original residual
thresholds. Roundoff Armijo branch always requires strictly decreasing residual.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_stable_transient import ImplicitFailure
from experiments.campaign_stable_transient_relative import RelativeStableTransient,RelativeState
from experiments.campaign_newton_stable_material import stable_material

def solve_inertial_roundoff(elastic,q_start,target,mass,force,mask,h_eff,tolerance=1e-10,maximum_iterations=20,energy_scale_J=1.):
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
            roundoff=128*np.finfo(float).eps*max(abs(V),abs(E),energy_scale_J)
            if VV<=V+1e-4*alpha*slope or(abs(VV-V)<=roundoff and np.linalg.norm(rr.ravel()[free])<norm):q=trial;history[-1]['alpha']=alpha;break
        else:raise ImplicitFailure('PositiveJ original-objective Armijo exhausted',history)
    raise AssertionError('Unreachable')


class RelativeStableTransientV03(RelativeStableTransient):
    def elastic_u(self,u,tangent=False):
        m=self.model;F=np.eye(3)+np.einsum('eai,eaJ->eiJ',u[m.tets],m.gradients);J=np.linalg.det(F)
        if not np.isfinite(J).all() or np.any(J<=.1):raise ValueError('SamepositiveJguard,norepair')
        W,P,A=stable_material(F,m.young,m.poisson,tangent);g=np.zeros_like(u);ge=np.einsum('e,eiJ,eaJ->eai',m.volumes,P,m.gradients);np.add.at(g,m.tets,ge);H=None
        if tangent:
            blocks=np.einsum('e,eaJ,eiJkL,ebL->eaibk',m.volumes,m.gradients,A,m.gradients,optimize=True)
            H=sparse.coo_matrix((blocks.ravel(),(m.rows,m.cols)),shape=(u.size,u.size)).tocsr()
        return float(m.volumes@W),g,H,J
    def step(self,state,h,method='BDF2',previous=None):
        if method not in ('BE','BDF2'):raise ValueError('Declaredmethodonly')
        if method=='BDF2' and previous is not None:heff=2*h/3;pred=state.delta/3+(2*h/9)*(4*state.v-previous.v);actual='BDF2'
        else:heff=h;pred=h*state.v;actual='BE_STARTUP' if method=='BDF2' else'BE'
        pred[self.mask]=0
        def elastic_increment(z,tangent=False):return self.elastic_u(state.u+z,tangent)
        scale=float(self.model.young*self.model.volumes.sum())
        z,d=solve_inertial_roundoff(elastic_increment,np.zeros_like(state.u),pred,self.mass,self.force,self.mask,heff,energy_scale_J=scale)
        v=(3*z-state.delta)/(2*h) if actual=='BDF2' else z/h;v[self.mask]=0
        acc=(3*v-4*state.v+previous.v)/(2*h) if actual=='BDF2' else(v-state.v)/h
        physical=self.mass[:,None]*acc+d['elastic_gradient_N']-self.force;u=state.u+z;world=self.model.rest+u
        d.update(actual_method=actual,h_eff_s=heff,physical_original_free_R_N=float(np.linalg.norm(physical[~self.mask])),constraint_defect_m=float(abs(u[self.mask]).max()),maximum_world_displacement_roundtrip_m=float(abs((world-self.model.rest)-u).max()),maximum_u_chord_roundoff_m=float(abs((u-state.u)-z).max()),maximum_world_chord_roundoff_m=float(abs((world-(self.model.rest+state.u))-z).max()),Armijo_roundoff_scale_J=scale,maximum_Armijo_roundoff_window_J=128*np.finfo(float).eps*scale)
        d.pop('elastic_gradient_N');return RelativeState(u,v,z),d
