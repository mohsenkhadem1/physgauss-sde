"""Observed-only nonlinear proposal transport; the physical posterior is unchanged.

A quadratic parameter regression centers native bridge states. A fixed dense
residual root and a smooth positive scalar scale whiten their remaining spread.
Every new state-map determinant is retained in the exact posterior target.
This experiment is separate from the immutable dense-reference full run.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from bird_local import load_model,make_target,sha,write,DeviceRenderer,gpu_check
from pacnerf_conditioned_transport import ConditionedTransport
from pacnerf_discrepancy import DiscrepancyTarget
from pacnerf_hybrid import conditional_reference_transform
from pacnerf_reflected_hybrid import ReflectedStateContext
from pacnerf_elliptical import atomic_save
from compare_pacnerf_discrepancy import replay


def features(v):
    result=[torch.ones_like(v[:,:1]),v]
    result += [v[:,i:i+1]*v[:,j:j+1] for i in range(v.shape[1]) for j in range(i,v.shape[1])]
    return torch.cat(result,-1)


class ConditionalBridge(ConditionedTransport):
    def __init__(self,model,reference):
        super().__init__(model,reference['anchor'],reference['guide'])
        self.learned=reference['conditional_bridge']
        self.indices=[0,1,*range(model.core_dimension,model.dimension)]

    def correction(self,z):
        p=self.learned
        standardized=(z[:,self.indices]-p['parameter_center'])/p['parameter_sd']
        design=features(standardized)
        mean=design@p['mean_coefficients']
        # Globally positive, bounded and differentiable. This bounds only
        # the proposal coordinates, never a physical state/noise parameter.
        log_scale=2*torch.tanh((design@p['log_scale_coefficients'])/2)
        return mean,log_scale

    def transform(self,z,inverse=False):
        p=self.learned;mean,log_scale=self.correction(z)
        if inverse:
            native,jac=super().transform(z,inverse=True)
            state=torch.linalg.solve_triangular(p['residual_root'],
                ((native[:,2:-3]-mean)/log_scale.exp()[:,None]).T,upper=False).T
            result=torch.cat((native[:,:2],state,native[:,-3:]),-1)
        else:
            state=mean+log_scale.exp()[:,None]*(z[:,2:-3]@p['residual_root'].T)
            native=torch.cat((z[:,:2],state,z[:,-3:]),-1)
            result,jac=super().transform(native)
        logdet=p['residual_root'].diagonal().log().sum()+mean.shape[-1]*log_scale
        return result,jac+logdet


class ConditionalBridgeContext(ReflectedStateContext):
    @torch.no_grad()
    def geometry(self):
        model=self.model;m=model.modes;core=self.core
        native=self.target.coordinates(self.fixed)
        _,log_scale=self.target.model.correction(native)
        correction=self.target.model.learned['residual_root']
        state_map=torch.linalg.solve_triangular(self.context['gaussian'][1].transpose(-1,-2),
            (correction@self.target.transform[2:core,2:core])[None].expand(len(self.fixed),-1,-1),upper=True)
        state_map=state_map*log_scale.exp()[:,None,None]
        q_index=torch.arange(m,model.base_dimension-2).reshape(13,2*m)[:,:m].flatten()
        q_map=(state_map[:,q_index]*model.scale[2+q_index][None,:,None]).reshape(len(self.fixed),13,m,core-2)
        normals=torch.einsum('gm,btms->btgs',model.floor_basis,q_map).flatten(1,2)
        zero=self.fixed.clone();zero[:,2:core]=0
        physical=self.target.model.to_physical(self.target.coordinates(zero))
        return normals,model.floor_heights(physical).flatten(1)


def conditional_target(model,reference):
    guide=ConditionalBridge(model,reference)
    target=DiscrepancyTarget(guide,reference['mean'],reference['precision'])
    target.transform=conditional_reference_transform(reference['precision'],model.core_dimension)
    target.reflection_context=lambda fixed:ConditionalBridgeContext(target,fixed,model.core_dimension)
    return target


@torch.no_grad()
def learn_reference(model,parent,training,regression_ridge=10.,covariance_shrinkage=.02):
    old=make_target(model,parent);core=model.core_dimension;d=model.dimension
    native=old.coordinates(training['white'].flatten(0,1))
    indices=[0,1,*range(core,d)];parameter=native[:,indices];state=native[:,2:core]
    center=parameter.mean(0);sd=parameter.std(0).clamp_min(.001)
    design=features((parameter-center)/sd)
    penalty=torch.eye(design.shape[-1],dtype=native.dtype)*regression_ridge;penalty[0,0]=0
    gram=design.T@design+penalty
    beta=torch.linalg.solve(gram,design.T@state)
    residual=state-design@beta
    covariance=torch.cov(residual.T)
    covariance=(1-covariance_shrinkage)*covariance+covariance_shrinkage*torch.diag(covariance.diag())
    covariance+=.001*torch.eye(core-2,dtype=native.dtype)
    root=torch.linalg.cholesky(covariance)
    standardized=torch.linalg.solve_triangular(root,residual.T,upper=False).T
    log_rms=.5*standardized.square().mean(-1).clamp_min(1e-12).log()
    scale_beta=torch.linalg.solve(gram,design.T@log_rms)
    parameter_covariance=torch.cov(parameter.T)+.001*torch.eye(len(indices),dtype=native.dtype)
    precision=torch.eye(d,dtype=native.dtype)
    ix=torch.tensor(indices)
    precision[ix[:,None],ix[None,:]]=torch.linalg.inv(parameter_covariance)
    mean=torch.zeros(d,dtype=native.dtype);mean[ix]=center
    reference=copy.deepcopy(parent)
    for key in ['sampling_mean','sampling_precision']:reference.pop(key,None)
    reference.update(mean=mean,precision=precision,reference_kind='conditional-quadratic-bridge-v1',
        conditional_bridge={'parameter_center':center,'parameter_sd':sd,
            'mean_coefficients':beta,'log_scale_coefficients':scale_beta,'residual_root':root})
    guide=ConditionalBridge(model,reference)
    endpoints=old.decode(training['white'][:,-1])
    new_native,_=guide.transform(endpoints,inverse=True)
    affine=conditional_reference_transform(precision,core)
    reference['initial_white']=torch.linalg.solve(affine,(new_native-mean).T).T
    new=conditional_target(model,reference)
    error=float((new.decode(reference['initial_white'])-endpoints).abs().max())
    if error>1e-8:raise ValueError('Conditional transport changed physical starting states')
    reference['proposal_training']={'discarded_draws':len(native),'features':design.shape[-1],
        'regression_ridge':regression_ridge,'covariance_shrinkage':covariance_shrinkage,
        'native_state_rmse_before_regression':float(state.std(0).square().mean().sqrt()),
        'native_state_rmse_after_regression':float(residual.square().mean().sqrt()),
        'log_scale_bounds':[-2.,2.],'maximum_initial_physical_state_error':error,
        'future_geometry_used':False,'physical_target_changed':False}
    return reference


def prepare(args):
    out=Path(args.output)
    if out.exists():raise ValueError('Use a fresh reference directory')
    model=load_model(args.inputs,json.loads(Path(args.gate).read_text()))
    parent=torch.load(args.reference,weights_only=True)
    training=torch.load(Path(args.gate).parent/'merged.pt',weights_only=True)
    old=make_target(model,parent);proof=replay(model,old,training)
    reference=learn_reference(model,parent,training)
    reference['proposal_training'].update(parent_density_replay=proof,
        parent_reference_sha256=sha(args.reference),parent_merged_sha256=sha(Path(args.gate).parent/'merged.pt'),
        source_sha256=sha(__file__))
    out.mkdir(parents=True);atomic_save(out/'reference.pt',reference)
    write(out/'proposal_training.json',reference['proposal_training'])
    print(json.dumps(reference['proposal_training']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',default='runs/inputs')
    p.add_argument('--gate',default='runs/bird-8000/convergence.json')
    p.add_argument('--reference',default='runs/reference-original/reference.pt')
    p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    prepare(args)
