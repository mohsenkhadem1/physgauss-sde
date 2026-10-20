"""Analytic conditional Gaussian STATE guide, with floor handled by slices.

Only the five-parameter reference covariance is learned. State coordinates have
unit reference covariance in the floor-free Gaussian bridge. The physical model
and its complete soft-floor density remain unchanged and Jacobian-corrected.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from pacnerf_conditioned_transport import ConditionedTransport,make_target
from pacnerf_hybrid import conditional_reference_transform
from pacnerf_elliptical import atomic_save


@torch.no_grad()
def make_reference(model,parent,training,parameter_ridge=.01):
    old=make_target(model,parent);dimension=model.dimension;core=model.core_dimension
    psi=torch.tensor([0,1,*range(core,dimension)])
    # Native parameter rows are independent of state-reference coordinates.
    parameter=old.mean[psi]+training['white'][...,psi]@old.transform[psi[:,None],psi[None,:]].T
    parameter=parameter.flatten(0,1)
    covariance=torch.cov(parameter.T)+parameter_ridge*torch.eye(len(psi),dtype=parameter.dtype)
    precision=torch.eye(dimension,dtype=parameter.dtype)
    precision[psi[:,None],psi[None,:]]=torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    mean=torch.zeros(dimension,dtype=parameter.dtype);mean[psi]=parameter.mean(0)
    guide=copy.deepcopy(parent['guide'])
    guide.pop('constraint_precision',None);guide.pop('constraint_rhs',None)
    reference={'coordinates':'floor-slice-gaussian-state-bridge-v1','mean':mean,'precision':precision,
        'guide':guide,'anchor':parent['anchor'].clone(),'parameter_blocked_reference':True,
        'frozen_model_specification':model.specification(),
        'reference_kind':'analytic-floor-free-state-gaussian-with-five-parameter-moments',
        'proposal_training':{'discarded_parent_draws':len(parameter),'parameter_ridge':parameter_ridge,
            'parameter_mean':mean[psi].tolist(),'parameter_covariance':covariance.tolist(),
            'state_covariance':'identity in the parameter-dependent Gaussian state bridge; no empirical state covariance',
            'future_geometry_used':False,'reference_is_posterior_mode':False}}
    physical=old.decode(training['position'])
    new_guide=ConditionedTransport(model,reference['anchor'],reference['guide'])
    native,_=new_guide.transform(physical,inverse=True)
    transform=conditional_reference_transform(precision,core)
    initial=torch.linalg.solve(transform,(native-mean).T).T
    reference['initial_white']=initial
    target=make_target(model,reference)
    reconstructed=target.decode(initial)
    error=float((reconstructed-physical).abs().max())
    if error>1e-9:raise ValueError('Floor-free proposal changed the initial physical states')
    reference['proposal_training']['maximum_physical_state_preservation_error']=error
    return reference


def prepare(args):
    from compare_pacnerf_discrepancy import make_inference_model,replay
    from pacnerf_recovery import restore_correlations
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    model,_=make_inference_model(args.inputs,8)
    settings=json.loads(Path(args.settings).read_text());restore_correlations(model,settings['model'])
    states=Path(args.states);parent=torch.load(states/'reference.pt',weights_only=True)
    training=torch.load(states/'chains.pt',weights_only=True)
    proof=replay(model,make_target(model,parent),training)
    reference=make_reference(model,parent,training)
    reference['proposal_training'].update({'parent_reference_sha256':hashlib.sha256((states/'reference.pt').read_bytes()).hexdigest(),
        'parent_checkpoint_sha256':hashlib.sha256((states/'chains.pt').read_bytes()).hexdigest(),
        'input_sha256':settings['input_sha256'],'original_target_replay':proof})
    output=Path(args.output)
    if output.exists():raise ValueError('Use a fresh proposal-reference directory')
    output.mkdir(parents=True);atomic_save(output/'reference.pt',reference)
    (output/'proposal_training.json').write_text(json.dumps(reference['proposal_training'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps(reference['proposal_training']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','states','settings','output']:p.add_argument('--'+name,required=True)
    prepare(p.parse_args())
