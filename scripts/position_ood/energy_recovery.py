"""Energy minimization warm start followed by an unmodified residual certificate.

The nonlinear potential and tangent are unchanged. L-BFGS only supplies an
initial point for Newton; its optimizer stopping status never certifies a field.
"""
import numpy as np
from scipy.optimize import minimize
from scripts.phystwin.consistent_fixture import ConsistentFixture
from scripts.position_ood.material_field import MaterialPositionFixture

class EnergyRecoveryFixture(MaterialPositionFixture):
    def solve(self,initial=None,maximum_iterations=40,force_tolerance_N=1e-9):
        try:return ConsistentFixture.solve(self,initial,maximum_iterations,force_tolerance_N)
        except RuntimeError as error:print('ENERGY_WARM_START',self.n,str(error),flush=True)
        self.clear_linear_cache();u=np.zeros_like(self.rest) if initial is None else np.asarray(initial).copy()
        state=self.evaluate(u,True);diagonal=state['tangent'].diagonal();scale=np.sqrt(np.maximum(abs(diagonal),max(abs(diagonal).max()*1e-8,1e-8)));del state
        best=dict(energy=float('inf'),u=u.copy());count=0
        def objective(v):
            nonlocal count
            trial=(v/scale).reshape(u.shape)
            try:s=self.evaluate(trial)
            except ValueError:return 1e100,np.zeros_like(v)
            count+=1
            if s['potential_J']<best['energy']:best.update(energy=s['potential_J'],u=trial.copy())
            if count%50==0:print('ENERGY_ITERATION',self.n,count,s['potential_J'],float(np.linalg.norm(s['residual'])),flush=True)
            return s['potential_J'],s['residual'].ravel()/scale
        result=minimize(objective,u.ravel()*scale,jac=True,method='L-BFGS-B',options=dict(maxiter=600,maxls=40,ftol=1e-15,gtol=1e-12,maxcor=20))
        print('ENERGY_WARM_FINISHED',self.n,result.message,count,flush=True);self.clear_linear_cache()
        state=ConsistentFixture.solve(self,best['u'],maximum_iterations,force_tolerance_N);state['energy_warm_evaluations']=count;return state
