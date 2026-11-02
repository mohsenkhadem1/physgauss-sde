"""Bounded conditional MAP over 15 continuous scene descriptors; not a posterior.
No scene file or target reader. Public geometry and categorical candidates only.
Parameter order: bulk log common scale/shape/heterogeneity amplitude (dimensionless);
left/right log K (K in N/m) and nonlinear strength; two material-coordinate
centers per support; two log dimensionless widths per support. Positions and
pose translations are metres, rotations radians, forces N, torques N m.
Noise standard deviations use these physical units. Priors are on this pack.
"""
import time
import copy
from dataclasses import asdict
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.compositional_neural_mechanics.scenes import make_model, material_field
from scripts.compositional_neural_mechanics.inference import implicit_gradient, observation_matrix

NAMES=('logscale','shape','gradient','left_logK','left_strength','right_logK','right_strength','left_center_y','left_center_z','right_center_y','right_center_z','left_logwidth_y','left_logwidth_z','right_logwidth_y','right_logwidth_z')

def neutral_parameters():
    return np.array([0.,0.,0.,np.log(500.),.5,np.log(500.),.5,.5,.5,.5,.5,*([np.log(.3)]*4)])

def default_bounds():
    return [(-1.,1.),(-2.,2.),(-.8,.8),(np.log(20.),np.log(10000.)),(0.,1.),(np.log(20.),np.log(10000.)),(0.,1.)]+[(.05,.95)]*4+[(np.log(.1),np.log(.6))]*4

class JointMAP:
    def __init__(self,geometry,*,bulk_family,interface_family,heterogeneity,observations,position_sigma,force_sigma=None,torque_sigma=None,prior_mean=None,prior_std=None,n=1,order=3,bulk_law=None,interface_law=None,device="cpu",gravity=(0.,0.,0.),density=125.,continuation=(.25,.5,.75,1.)):
        if device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only CPU/GPU0–2 allowed')
        gravity=np.asarray(gravity,float);fractions=np.asarray(continuation,float)
        if gravity.shape!=(3,) or not np.isfinite(gravity).all() or not np.isfinite(density) or density<=0:raise ValueError('Finite known gravity/positive density required')
        if fractions.ndim!=1 or len(fractions)==0 or not np.isfinite(fractions).all() or fractions[0]<=0 or fractions[-1]!=1 or (np.diff(fractions)<=0).any():raise ValueError('Increasing continuation fractions ending at1 required')
        self.device=device;self.continuation=fractions.copy();self._observation_cache={};self._deadline=None
        if bulk_family not in ('mooney_rivlin','yeoh') or interface_family not in (0,1) or heterogeneity not in (0,1):raise ValueError('Explicit categorical candidates required')
        if position_sigma<=0 or not np.isfinite(position_sigma):raise ValueError('Positive position noise required')
        self.mean=neutral_parameters() if prior_mean is None else np.asarray(prior_mean,float).copy()
        self.std=np.ones(15) if prior_std is None else np.asarray(prior_std,float).copy()
        if self.mean.shape!=(15,) or self.std.shape!=(15,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():raise ValueError('Finite explicit 15D prior required')
        self.scene=dict(geometry=asdict(geometry),bulk_family=bulk_family,bulk_latent=[0.,0.,-1. if bulk_family=='mooney_rivlin' else 1.],gradient=0.,factors={'heterogeneity':heterogeneity},interface_latents=[[np.log(500.),.5,float(interface_family)]]*2,centers=[[.5,.5]]*2,widths=[[.3,.3]]*2,gravity=gravity.tolist(),density_kg_m3=float(density))
        # Own device-local frozen copies; caller checkpoints/weights remain untouched.
        def local_law(law):
            if law is None:return None
            result=copy.deepcopy(law).to(device=device,dtype=torch.float64)
            for parameter in result.parameters():parameter.requires_grad_(False)
            return result
        self.kw=dict(n=n,order=order,bulk_law=local_law(bulk_law),interface_law=local_law(interface_law),device=device)
        self.sigmas=(position_sigma,force_sigma,torque_sigma);self.observations=[]
        for record in observations:
            o={k:np.array(v,dtype=float,copy=True) for k,v in record.items()}
            if set(o)-{'pose','points','displacement','forces','torques'}:raise ValueError('Unknown observation channels')
            if o['pose'].shape!=(2,6) or o['points'].ndim!=2 or o['points'].shape[1]!=3 or o['displacement'].shape!=o['points'].shape:raise ValueError('Observation dimensions')
            for key,sigma in [('forces',force_sigma),('torques',torque_sigma)]:
                if key in o and (o[key].shape!=(2,3) or sigma is None or not np.isfinite(sigma) or sigma<=0):raise ValueError('Wrench observations require declared noise')
            if not all(np.isfinite(v).all() for v in o.values()):raise ValueError('Finite observations required')
            if len(o['points'])==0 or (o['points']<0).any() or (o['points']>1).any():raise ValueError('Material points must be inside unit reference cube')
            self.observations.append(o)
        if not self.observations:raise ValueError('At least one observation required')

    def bind(self,model,z):
        bulk=torch.stack([z[0],z[1],z.new_tensor(self.scene['bulk_latent'][2]),z[2]])
        model.material_latent_field=material_field(self.scene,bulk)
        model.interface_latent_fields=tuple((lambda q,side=side:torch.stack([z[3+2*side],z[4+2*side],z.new_tensor(self.scene['interface_latents'][side][2])]).expand(*q.shape[:-1],3)) for side in range(2))
        model.interface_geometry_parameters=lambda:torch.stack([torch.cat([z[7:9],z[11:13]]),torch.cat([z[9:11],z[13:15]])])

    def value_gradient(self,parameters):
        z=torch.tensor(np.asarray(parameters),dtype=torch.float64,device=self.device)
        if z.shape!=(15,) or not torch.isfinite(z).all():raise ValueError('Finite15D parameters required')
        prior=(z-z.new_tensor(self.mean))/z.new_tensor(self.std)
        value=float(.5*prior.square().sum());gradient=((z-z.new_tensor(self.mean))/z.new_tensor(self.std).square()).cpu().numpy();diagnostics=[]
        for o in self.observations:
            model=make_model(self.scene,**self.kw);self.bind(model,z)
            initial=None
            for fraction in self.continuation:
                if self._deadline is not None and time.monotonic()>=self._deadline:raise TimeoutError('MAP continuation budget exhausted')
                model.set_controls(o['pose']*fraction)
                solved=model.solve(initial=initial,force_tolerance_N=1e-10,maximum_iterations=60)
                initial=solved['displacement_local_m']
            u=initial
            key=(o['points'].shape,o['points'].tobytes())
            if key not in self._observation_cache:self._observation_cache[key]=observation_matrix(model,o['points'])
            M=self._observation_cache[key]
            H=model.evaluate(u,True)['tangent']
            def residual(u,live):
                self.bind(model,live);return model.torch_residual(u)
            def objective(u,live):
                self.bind(model,live)
                error=(u.new_tensor(M)@u-u.new_tensor(o['displacement']))/self.sigmas[0]
                loss=.5*error.square().sum()
                if 'forces' in o or 'torques' in o:
                    _,_,_,(forces,torques)=model.interface_response(u,create_graph=True)
                    for key,pred,sigma in [('forces',forces,self.sigmas[1]),('torques',torques,self.sigmas[2])]:
                        if key in o:loss=loss+.5*((torch.stack(pred)-u.new_tensor(o[key]))/sigma).square().sum()
                return loss
            result=implicit_gradient(u,z,residual,objective,tangent=H,equilibrium_tolerance=1e-8)
            value+=result['loss'];gradient+=result['gradient'].cpu().numpy()
            diagnostics.append({k:result[k] for k in ('equilibrium_residual_norm','adjoint_relative_residual','tangent_direction_relative_error')})
        return value,gradient,diagnostics

    def fit(self,*,active,initial=None,bounds=None,max_iterations=20,max_evaluations=60,seconds=120):
        active=np.asarray(active,int)
        if active.ndim!=1 or len(active)==0 or len(set(active.tolist()))!=len(active) or (active<0).any() or (active>=15).any():raise ValueError('Distinct parameter subset required')
        base=self.mean.copy() if initial is None else np.asarray(initial,float).copy()
        limits=default_bounds() if bounds is None else bounds
        if base.shape!=(15,) or len(limits)!=15 or seconds<=0 or max_iterations<1 or max_evaluations<1:raise ValueError('Invalid fit budget/config')
        if not np.isfinite(base).all() or any(len(b)!=2 or not np.isfinite(b).all() or b[0]>=b[1] for b in limits):raise ValueError('Finite ordered bounds required')
        if any(not limits[i][0]<=base[i]<=limits[i][1] for i in range(15)):raise ValueError('Initial parameters outside bounds')
        started=time.monotonic();history=[];self._deadline=started+seconds
        def objective(x):
            if time.monotonic()-started>=seconds or len(history)>=max_evaluations:raise TimeoutError('MAP fit budget exhausted')
            z=base.copy();z[active]=x
            value,grad,diag=self.value_gradient(z)
            history.append(dict(loss=value,parameters=z.tolist(),diagnostics=diag))
            if time.monotonic()-started>=seconds:raise TimeoutError('MAP fit budget exhausted')
            return value,grad[active]
        try:
            result=minimize(objective,base[active],jac=True,method='L-BFGS-B',bounds=[limits[i] for i in active],options={'maxiter':max_iterations,'maxfun':max_evaluations,'ftol':1e-10,'gtol':1e-7})
        except TimeoutError as error:
            best=min(history,key=lambda row:row['loss']) if history else None
            return dict(parameters=np.array(best['parameters']) if best else base.copy(),loss=best['loss'] if best else None,optimizer_success=False,incomplete=True,message=str(error),evaluations=history,elapsed_seconds=time.monotonic()-started,scope='Budget-exhausted conditional MAP; best complete evaluation only; not converged')
        finally:self._deadline=None
        fitted=base.copy();fitted[active]=result.x
        return dict(parameters=fitted,loss=float(result.fun),optimizer_success=bool(result.success),incomplete=not bool(result.success),message=str(result.message),evaluations=history,elapsed_seconds=time.monotonic()-started,scope='Conditional bounded MAP; categorical candidates supplied; no posterior/identifiability claim')
