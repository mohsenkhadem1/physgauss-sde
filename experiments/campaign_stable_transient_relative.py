"""Numerical reparameterization of frozen BE/BDF2, same energy and tolerances.

Solve step increment z, retain displacement u/physical v. Elastic energy remains
EXACT frozen model.elastic(rest+u). World output/chord rounding is explicit.
"""
from dataclasses import dataclass
import numpy as np
from experiments.campaign_stable_transient import StableBlockTransient,solve_inertial

@dataclass
class RelativeState:
    u:np.ndarray
    v:np.ndarray
    delta:np.ndarray

class RelativeStableTransient(StableBlockTransient):
    def initial(self):
        zero=np.zeros_like(self.model.rest);return RelativeState(zero.copy(),zero.copy(),zero.copy())
    def step(self,state,h,method='BDF2',previous=None):
        if method not in ('BE','BDF2'):raise ValueError('Same declared BE/BDF2 required')
        if method=='BDF2' and previous is not None:
            heff=2*h/3;pred=state.delta/3+(2*h/9)*(4*state.v-previous.v);actual='BDF2'
        else:heff=h;pred=h*state.v;actual='BE_STARTUP' if method=='BDF2' else'BE'
        pred[self.mask]=0
        def elastic_increment(z,tangent=False):return self.model.elastic(self.model.rest+(state.u+z),tangent=tangent)
        z,d=solve_inertial(elastic_increment,np.zeros_like(state.u),pred,self.mass,self.force,self.mask,heff)
        v=(3*z-state.delta)/(2*h) if actual=='BDF2' else z/h;v[self.mask]=0
        acc=(3*v-4*state.v+previous.v)/(2*h) if actual=='BDF2' else(v-state.v)/h
        physical=self.mass[:,None]*acc+d['elastic_gradient_N']-self.force;u=state.u+z;world=self.model.rest+u
        d.update(actual_method=actual,h_eff_s=heff,physical_original_free_R_N=float(np.linalg.norm(physical[~self.mask])),constraint_defect_m=float(abs(u[self.mask]).max()),maximum_world_displacement_roundtrip_m=float(abs((world-self.model.rest)-u).max()),maximum_u_chord_roundoff_m=float(abs((u-state.u)-z).max()),maximum_world_chord_roundoff_m=float(abs((world-(self.model.rest+state.u))-z).max()))
        d.pop('elastic_gradient_N');return RelativeState(u,v,z),d
