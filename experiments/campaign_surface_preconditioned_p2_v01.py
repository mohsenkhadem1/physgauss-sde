"""Same P2 potential/Newton acceptance with verified preconditioned directions.

No physics executes on import. Frozen direct-LU references are unchanged.
An exact rest factor may serve linear response and both quadrature targets;
the two nonlinear states/line searches are always solved independently.
"""
import numpy as np
from experiments.campaign_surface_p2_static_v02 import P2NonlinearReference
from experiments.campaign_static_uniaxial_bar import StaticFailure
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG


class PreconditionedP2Reference(P2NonlinearReference):
    @classmethod
    def from_forward(cls,forward,E,K,*,quadrature_order=24,direction_solver=None):
        self=super().from_forward(forward,E,K,quadrature_order=quadrature_order)
        expected=E*forward.Kunit_free+K*forward.W_free
        self.direction_solver=ExactRestPCG(expected)if direction_solver is None else direction_solver
        difference=self.direction_solver.rest-expected
        if difference.shape!=expected.shape or np.max(abs(difference.data),initial=0)!=0:
            raise ValueError('Shared rest factor must match exact geometry and E/K')
        return self

    def solve(self,force,*,initial=None,maximum_iterations=40,force_tolerance_N=1e-10):
        self.load=np.asarray(force,float).copy()
        if self.load.shape!=self.rest.shape or not np.isfinite(self.load).all():raise ValueError('Finite explicit force required')
        u=np.zeros_like(self.rest)if initial is None else np.asarray(initial,float).copy()
        if u.shape!=self.rest.shape or not np.isfinite(u).all()or np.any(u[self.mask]!=0):raise ValueError('Initial state must obey exact clamps')
        if not np.isfinite(force_tolerance_N)or force_tolerance_N<=0:raise ValueError('Positive original-force tolerance required')
        tolerance=float(force_tolerance_N);history=[];directions=[]
        for iteration in range(maximum_iterations+1):
            state=self.evaluate(u,tangent=True);r=state['residual'].ravel()[self.free];norm=float(np.linalg.norm(r))
            history.append(dict(iteration=iteration,residual_N=norm,potential_J=state['potential_J'],minimum_J=float(state['J'].min())))
            if norm<=tolerance:
                return u,dict(iterations=iteration,history=history,minimum_J=float(state['J'].min()),
                    maximum_free_residual_N=float(np.max(abs(r))),sampled_J_only=True,
                    linear_directions=directions,unmodified_target_potential=True,
                    rest_factorizations=self.direction_solver.rest_factorizations)
            if iteration==maximum_iterations:raise StaticFailure('Static original-force residual exceeds cap',history)
            matrix=state['tangent'][self.free][:,self.free]
            direction,diagnostic=self.direction_solver.direction(matrix,-r);directions.append(diagnostic)
            slope=float(r@direction)
            if not np.isfinite(direction).all()or not np.isfinite(slope)or slope>=0:raise StaticFailure('Unmodified Newton direction not descent',history)
            # Identical Armijo and explicit force-decrease roundoff branch to
            # the frozen QuadraticBlock.solve; no changed objective/stiffness.
            for backtrack in range(40):
                alpha=2.**(-backtrack);trial=u.copy();trial.ravel()[self.free]+=alpha*direction
                try:candidate=self.evaluate(trial)
                except ValueError:continue
                roundoff=128*np.finfo(float).eps*max(abs(state['potential_J']),self.young*self.volumes.sum(),1e-30)
                if candidate['potential_J']<=state['potential_J']+1e-4*alpha*slope or(abs(candidate['potential_J']-state['potential_J'])<=roundoff and np.linalg.norm(candidate['residual'].ravel()[self.free])<norm):
                    u=trial;history[-1]['alpha']=alpha;break
            else:raise StaticFailure('PositiveJ static Armijo exhausted',history)
        raise AssertionError('Unreachable static solve')
