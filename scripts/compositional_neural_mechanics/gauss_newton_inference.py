"""Scaled trust-region least squares for the unchanged JointMAP objective.

This module changes optimization only. No posterior, categorical search or
identifiability claim. A process-level timeout must bound production runs; the
internal clock is checked between solver calls and derivative directions.
This implementation assumes all displacement DOFs are free (bilateral supports);
future essential constraints require a reduced free-DOF solve.
"""
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.inference import observation_matrix
from scripts.compositional_neural_mechanics.joint_inference import default_bounds


class GaussNewtonMAP:
    def __init__(self,estimator):
        self.estimator=estimator;self.deadline=None;self.observation_cache={}

    def check_budget(self):
        if self.deadline is not None and time.monotonic()>=self.deadline:raise TimeoutError('Gauss-Newton time budget exhausted')

    def residual_jacobian(self,parameters,active=None):
        e=self.estimator;values=np.asarray(parameters,float)
        if values.shape!=(15,) or not np.isfinite(values).all():raise ValueError('Finite15D parameters required')
        active=np.arange(15) if active is None else np.asarray(active,int)
        if active.ndim!=1 or not len(active) or len(set(active.tolist()))!=len(active) or (active<0).any() or (active>=15).any():raise ValueError('Distinct active indices required')
        z=torch.tensor(values,dtype=torch.float64,device=e.device);blocks=[];jacobians=[];diagnostics=[]
        for o in e.observations:
            self.check_budget();model=make_model(e.scene,**e.kw);e.bind(model,z);u=None
            for fraction in e.continuation:
                self.check_budget();model.set_controls(o['pose']*fraction)
                state=model.solve(initial=u,force_tolerance_N=1e-10,maximum_iterations=60);u=state['displacement_local_m']
            H=model.evaluate(u,True)['tangent'];factor=splu(H.tocsc());ut=torch.tensor(u,dtype=torch.float64,device=e.device)
            def residual(live):e.bind(model,live);return model.torch_residual(ut)
            columns=[]
            for index in active:
                self.check_budget();direction=torch.zeros_like(z);direction[index]=1
                _,derivative=torch.autograd.functional.jvp(residual,z,direction,create_graph=False,strict=False)
                columns.append(derivative.detach().cpu().numpy().ravel())
            Rz=np.stack(columns,1);du=factor.solve(-Rz)
            key=(o['points'].shape,o['points'].tobytes())
            if key not in self.observation_cache:self.observation_cache[key]=observation_matrix(model,o['points'])
            M=self.observation_cache[key];blocks.append(((M@u-o['displacement'])/e.sigmas[0]).ravel())
            jacobians.append(np.einsum('pn,ncj->pcj',M,du.reshape(-1,3,len(active))).reshape(-1,len(active))/e.sigmas[0])
            if 'forces' in o or 'torques' in o:
                def wrench(displacement,live):
                    e.bind(model,live);_,_,_,(f,t)=model.interface_response(displacement,create_graph=True)
                    return torch.cat([torch.stack(pred).reshape(-1)/sigma for key,pred,sigma in [('forces',f,e.sigmas[1]),('torques',t,e.sigmas[2])] if key in o])
                predicted=wrench(ut,z).detach().cpu().numpy()
                target=np.concatenate([o[key].ravel()/sigma for key,sigma in [('forces',e.sigmas[1]),('torques',e.sigmas[2])] if key in o]);blocks.append(predicted-target)
                derivatives=[]
                for col,index in enumerate(active):
                    self.check_budget();direction=torch.zeros_like(z);direction[index]=1
                    displacement_direction=torch.tensor(du[:,col].reshape(u.shape),dtype=torch.float64,device=e.device)
                    _,d=torch.autograd.functional.jvp(wrench,(ut,z),(displacement_direction,direction),create_graph=False,strict=False)
                    derivatives.append(d.detach().cpu().numpy())
                jacobians.append(np.stack(derivatives,1))
            e.bind(model,z)
            live_u=ut.detach().requires_grad_(True);live_r=model.torch_residual(live_u)
            direction=torch.sin(torch.arange(live_u.numel(),dtype=live_u.dtype,device=live_u.device)+1).reshape_as(live_u)
            autodiff=torch.autograd.grad((live_r*direction).sum(),live_u)[0].detach().cpu().numpy().ravel()
            assembled=H.T@direction.detach().cpu().numpy().ravel()
            diag=dict(equilibrium_residual_norm=float(np.linalg.norm(state['residual'])),linear_relative_residual=float(np.linalg.norm(H@du+Rz)/max(1e-30,np.linalg.norm(Rz))),
                      tangent_direction_relative_error=float(np.linalg.norm(autodiff-assembled)/max(1e-30,np.linalg.norm(autodiff))))
            if not all(np.isfinite(v) and v<=1e-8 for v in diag.values()):raise RuntimeError(f'Implicit derivative qualification failed: {diag}')
            diagnostics.append(diag)
        blocks.append((values-e.mean)/e.std);jacobians.append(np.diag(1/e.std)[:,active])
        r=np.concatenate(blocks);J=np.concatenate(jacobians)
        if not np.isfinite(r).all() or not np.isfinite(J).all():raise ValueError('Nonfinite least-squares residual/Jacobian')
        return r,J,diagnostics

    def fit(self,*,active=None,initial=None,bounds=None,max_evaluations=40,seconds=900):
        e=self.estimator;active=np.arange(15) if active is None else np.asarray(active,int)
        if active.ndim!=1 or not len(active) or len(set(active.tolist()))!=len(active) or (active<0).any() or (active>=15).any():raise ValueError('Distinct active indices required')
        base=e.mean.copy() if initial is None else np.asarray(initial,float).copy();limits=np.asarray(default_bounds() if bounds is None else bounds,float)
        if base.shape!=(15,) or limits.shape!=(15,2) or not np.isfinite(base).all() or not np.isfinite(limits).all() or (limits[:,0]>=limits[:,1]).any() or (base<limits[:,0]).any() or (base>limits[:,1]).any():raise ValueError('Finite initial parameters within declared bounds required')
        if seconds<=0 or not np.isfinite(seconds) or max_evaluations<1:raise ValueError('Positive bounded budget required')
        started=time.monotonic();self.deadline=started+seconds;history=[];cache={}
        def evaluate(x):
            self.check_budget()
            if 'x' in cache and np.array_equal(cache['x'],x):return cache['r'],cache['J']
            if len(history)>=max_evaluations:raise TimeoutError('Gauss-Newton evaluation budget exhausted')
            full=base.copy();full[active]=x;r,J,diag=self.residual_jacobian(full,active)
            history.append(dict(parameters=full.tolist(),loss=float(.5*r@r),gradient_infinity=float(np.max(abs(J.T@r))),diagnostics=diag))
            cache.update(x=x.copy(),r=r,J=J);self.check_budget();return r,J
        try:
            result=least_squares(lambda x:evaluate(x)[0],base[active],jac=lambda x:evaluate(x)[1],bounds=(limits[active,0],limits[active,1]),method='trf',x_scale='jac',max_nfev=max_evaluations,ftol=1e-10,xtol=1e-10,gtol=1e-7)
            fitted=base.copy();fitted[active]=result.x
            out=dict(parameters=fitted,loss=float(result.cost),optimizer_success=bool(result.success),incomplete=not bool(result.success),message=str(result.message),optimality=float(result.optimality),scipy_nfev=int(result.nfev))
        except (TimeoutError,ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            best=min(history,key=lambda x:x['loss']) if history else None
            out=dict(parameters=np.asarray(best['parameters']) if best else base,loss=best['loss'] if best else None,optimizer_success=False,incomplete=True,message=f'{type(error).__name__}: {error}')
        finally:self.deadline=None
        out.update(evaluations=history,elapsed_seconds=time.monotonic()-started,scope='Same conditional JointMAP objective; trust-region reflective least squares with analytic implicit Jacobian and Jacobian column scaling; not a posterior')
        return out
