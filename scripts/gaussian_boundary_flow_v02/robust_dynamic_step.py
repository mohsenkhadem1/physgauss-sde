"""Versioned dynamic Newton target, separate from physical admission tolerance.

The original energy solve targets 1e-9 (midpoint 1e-10), but dynamic rollouts admit
residual <=1e-7. This candidate targets 1e-8: still ten times tighter than admission.
No failed old trajectory is retrospectively accepted. No residual gate, force,
energy, state, or Newton update is changed. Static solves retain their own target.
"""
import torch
from scripts.gaussian_boundary_flow_v02.finite_range_attachment import FiniteRangeAttachmentFEM


class DynamicSolveTargetMixin:
    dynamic_solver_target=1e-8

    def solve(self,theta,frame,initial=None,tolerance=1e-7,maxiter=80):
        effective=tolerance
        if self.inertial_target is not None:
            effective=max(tolerance,self.dynamic_solver_target)
        state,H,diag=super().solve(theta,frame,initial,tolerance=effective,maxiter=maxiter)
        return state,H,{**diag,'requested_solver_tolerance':tolerance,'effective_solver_tolerance':effective}


class FiniteRangeAttachmentFEMTargetV2(DynamicSolveTargetMixin,FiniteRangeAttachmentFEM):
    pass


class SafeguardedDynamicSolveMixin:
    """Target 1e-8; stop near roundoff only if force gate AND tiny correction pass."""
    @torch.no_grad()
    def solve(self,theta,frame,initial=None,tolerance=1e-7,maxiter=80):
        import torch
        if self.inertial_target is None:
            return super().solve(theta,frame,initial,tolerance=tolerance,maxiter=maxiter)
        x=self.rest.clone() if initial is None else initial.clone()
        target=max(tolerance,1e-8)
        for iteration in range(maxiter):
            E,r,H=self.evaluate(x,theta,frame,True);norm=float(torch.linalg.vector_norm(r))
            if not torch.isfinite(r).all():raise RuntimeError('Nonfinite force residual')
            correction=None
            try:
                direction=torch.linalg.solve(H,-r.flatten()).reshape_as(x)
                if bool(torch.isfinite(direction).all()):correction=float(direction.abs().max())
            except RuntimeError:pass
            safeguarded=norm<=1e-7 and correction is not None and correction<=1e-9
            if norm<target or safeguarded:
                return x,H,dict(iterations=iteration+1,residual=norm,minJ=self.min_jacobian(x),effective_solver_tolerance=target,safeguarded_termination=safeguarded and norm>=target,newton_correction_max=correction)
            diag=torch.diag(H).abs().mean().clamp_min(1e-5);accepted=False
            for damping in [0.,1e-5,1e-3,.1,1.,10.]:
                try:direction=torch.linalg.solve(H+damping*diag*torch.eye(self.ndof,device=self.device,dtype=self.dtype),-r.flatten()).reshape_as(x)
                except RuntimeError:continue
                slope=(direction*r).sum()
                if not torch.isfinite(direction).all() or float(slope)>=0:continue
                for j in range(22):
                    step=2.**(-j)
                    try:newE,newr,_=self.evaluate(x+step*direction,theta,frame)
                    except ValueError:continue
                    roundoff_refinement=(norm<=1e-7 and float(torch.linalg.vector_norm(newr))<.5*norm and float(newE)<=float(E)+64*torch.finfo(self.dtype).eps*max(1.,abs(float(E))))
                    if float(newE)<=float(E+1e-4*step*slope) or roundoff_refinement:
                        x=x+step*direction;accepted=True;break
                if accepted:break
            if not accepted:raise RuntimeError(f'Safeguarded dynamic line search failed residual={norm:g}, correction={correction}')
        raise RuntimeError('Safeguarded dynamic iteration limit')


class FiniteRangeAttachmentFEMSafeguardedV3(SafeguardedDynamicSolveMixin,FiniteRangeAttachmentFEM):
    pass
