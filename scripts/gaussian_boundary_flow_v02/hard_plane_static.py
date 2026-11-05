"""Isolated frictionless static hard-plane contact via constrained energy minimization.

min_x V(x), subject to x[:,2] >= floor. KKT: grad(V)-lambda*ez=0,
lambda>=0, gap>=0, lambda*gap=0. No soft normal penalty, positional gauge,
posthoc pose projection, inertial response, inferred friction or impact claim.
The inherited theta floor slot is ignored for layout compatibility ONLY; any
future inference adapter must remove it rather than infer an unobservable value.
"""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM


class StaticHardPlaneFEM(ExternalFEM):
    def __init__(self,*args,**kwargs):
        if kwargs.get('contact',False):raise ValueError('This candidate is frictionless; tangential penalty disabled')
        kwargs['contact']=False
        super().__init__(*args,**kwargs)
        if not np.isfinite(self.floor):raise ValueError('Known finite plane required')

    def evaluate(self,x,theta,frame,tangent=False):
        if self.inertial_target is not None:raise ValueError('Static hard-plane candidate has no dynamic qualification')
        parameters=theta.clone();parameters[self.i_floor]=-torch.inf
        E,r,H=super().evaluate(x,parameters,frame,tangent)
        delta=x-self.rest;pin=1e-5
        E=E-.5*pin*delta.square().sum();r=r-pin*delta
        if H is not None:H=H-pin*torch.eye(self.ndof,device=self.device,dtype=self.dtype)
        return E,r,H

    def kkt(self,x,residual,contact_tolerance=1e-10):
        gap=x[:,2]-self.floor;touch=gap<=contact_tolerance
        reaction=torch.where(touch,residual[:,2].clamp_min(0),torch.zeros_like(gap))
        stationarity=residual.clone();stationarity[:,2]-=reaction
        return {'stationarity_norm':float(torch.linalg.vector_norm(stationarity)),
                'minimum_gap':float(gap.min()),'maximum_penetration':float((-gap).clamp_min(0).max()),
                'minimum_reaction':float(reaction.min()),'complementarity_max':float((reaction*gap).abs().max()),
                'reaction':reaction,'touching':touch,'stationarity':stationarity}

    @torch.no_grad()
    def solve(self,theta,frame,initial=None,tolerance=1e-8,maxiter=100):
        if not 0<tolerance<=1e-7:raise ValueError('Static stationarity target must be <=1e-7')
        x=self.rest.clone()if initial is None else initial.clone()
        if not bool(torch.isfinite(x).all())or bool((x[:,2]<self.floor-1e-12).any()):
            raise ValueError('Finite feasible initial state required; no initial penetration projection')
        # Floating-point endpoint rounding only, never a returned-state correction.
        x[:,2]=x[:,2].clamp_min(self.floor)
        rejected=0;accepted_steps=0
        for iteration in range(maxiter+1):
            E,r,H=self.evaluate(x,theta,frame,True);kkt=self.kkt(x,r)
            if not bool(torch.isfinite(r).all()):raise RuntimeError('Nonfinite mechanical residual')
            if kkt['stationarity_norm']<=tolerance and kkt['maximum_penetration']<=1e-12 and kkt['complementarity_max']<=1e-10:
                diag={k:v for k,v in kkt.items()if not isinstance(v,torch.Tensor)}
                diag.update(iterations=iteration,accepted_steps=accepted_steps,rejected_trials=rejected,minJ=self.min_jacobian(x),energy=float(E),normal_reactions=kkt['reaction'].cpu().tolist(),residual=kkt['stationarity_norm'],raw_mechanical_state=True,posthoc_projection=False)
                return x,H,diag
            if iteration==maxiter:break
            active=kkt['touching']&(r[:,2]>=0)
            fixed=torch.zeros_like(x,dtype=torch.bool);fixed[:,2]=active
            free=~fixed.flatten();ids=torch.nonzero(free).flatten()
            Hfree=H[ids[:,None],ids[None,:]];rfree=r.flatten()[ids]
            curvature=float(torch.diag(Hfree).abs().mean().clamp_min(1e-6))
            directions=[]
            for damping in (0.,1e-8,1e-5,1e-3,.1,1.,10.):
                try:
                    step=torch.linalg.solve(Hfree+damping*curvature*torch.eye(len(ids),device=self.device,dtype=self.dtype),-rfree)
                except RuntimeError:continue
                direction=torch.zeros(self.ndof,device=self.device,dtype=self.dtype);direction[ids]=step
                directions.append(direction.reshape_as(x))
            directions.append(-kkt['stationarity']/curvature)
            accepted=False
            for direction in directions:
                # Feasible tangent cone; downward motions at a bound are disallowed.
                direction=direction.clone();direction[:,2]=torch.where(kkt['touching'],direction[:,2].clamp_min(0),direction[:,2])
                slope=float((r*direction).sum())
                if not bool(torch.isfinite(direction).all())or slope>=0:continue
                downward=direction[:,2]<0
                maximum_step=1.
                if bool(downward.any()):maximum_step=min(1.,float(((x[downward,2]-self.floor)/(-direction[downward,2])).min()))
                if maximum_step<=1e-14:continue
                for ls in range(24):
                    alpha=maximum_step*2.**(-ls);trial=x+alpha*direction
                    trial[:,2]=trial[:,2].clamp_min(self.floor) # Roundoff at exact bound only.
                    try:newE,newr,_=self.evaluate(trial,theta,frame)
                    except(ValueError,RuntimeError):rejected+=1;continue
                    newkkt=self.kkt(trial,newr)
                    roundoff=(kkt['stationarity_norm']<=1e-7 and newkkt['stationarity_norm']<.5*kkt['stationarity_norm'] and float(newE)<=float(E)+64*torch.finfo(self.dtype).eps*max(1.,abs(float(E))))
                    if float(newE)<=float(E)+1e-4*alpha*slope or roundoff:
                        x=trial;accepted=True;accepted_steps+=1;break
                    rejected+=1
                if accepted:break
            if not accepted:raise RuntimeError(f'Hard-plane line search failed; KKT residual={kkt["stationarity_norm"]:.9g}, minJ={self.min_jacobian(x):.9g}')
        raise RuntimeError(f'Hard-plane iteration limit; KKT residual={kkt["stationarity_norm"]:.9g}, minJ={self.min_jacobian(x):.9g}')
