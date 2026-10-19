"""Fit an observed-data-only Gaussian-bridge reference, not a physical MAP."""
import math
import time
import torch
import pacnerf_innovation_metric as old_fitting
from pacnerf_discrepancy import DiscrepancyInnovation,gauss_newton
from pacnerf_conditioned_transport import ConditionedTransport
from pacnerf_comparison import optimize,write,initialize,observed_hessian


def full_curvature_reference(objective,mean):
    began=time.time()
    hessian=observed_hessian(lambda z:objective(z[None]).sum(),mean,block_size=4)
    hessian=(hessian+hessian.T)/2
    eigen,vectors=torch.linalg.eigh(hessian)
    floor=max(.05,float(eigen.abs().max())*1e-7)
    precision=(vectors*eigen.clamp_min(floor))@vectors.T
    return precision,{'scope':'full observed conditioned-coordinate Hessian; fixed proposal only, never substituted for target',
        'minimum_observed_eigenvalue':float(eigen.min()),'maximum_observed_eigenvalue':float(eigen.max()),
        'proposal_eigenvalue_floor':floor,'adjusted_eigenvalues':int((eigen<floor).sum()),'seconds':time.time()-began}


def fit_reference(model,checkpoint=None,anchor=None):
    began=time.time();d=model.dimension
    if anchor is None and hasattr(model,'floor_penalty'):
        # A reference initialization, using known gravity/initial geometry and
        # the known plane only. No later particles or image truth are accessed.
        anchor=initialize(model)
        with torch.no_grad():
            height=model.floor_heights(anchor[None])[0]
            lift=(.01-height.amin(-1)).clamp_min(0)
            vertical=model.basis[:,1].sum(0)
            for t in range(13):
                start=2+model.modes+t*2*model.modes
                anchor[start:start+model.modes]+=lift[t]*vertical/.1
        prior_stationarity={'qualification':'Known-geometry ballistic floor-feasible initialization; not a fitted mode. Complete conditioned density is optimized and checked below.'}
        prior_info={'initialization':'gravity plus uniform upward modal projection to keep all initial path heights at least .01m'}
        initialized_floor=True
    else:initialized_floor=False
    if anchor is None:
        innovation=DiscrepancyInnovation(model)
        original=old_fitting.gauss_newton;old_fitting.gauss_newton=gauss_newton
        try:prior_mean,_,prior_stationarity,prior_info=old_fitting.fit_reference(innovation,checkpoint=checkpoint)
        finally:old_fitting.gauss_newton=original
        with torch.no_grad():anchor=innovation.to_physical(prior_mean[None])[0]
    elif not initialized_floor:prior_stationarity={'qualification':'provided observed-only development anchor'};prior_info={}
    indices=torch.tensor([0,1,d-3,d-2,d-1])
    parameters=anchor[indices].detach().clone()
    history=[]
    for round_ in range(2):
        guide=ConditionedTransport(model,anchor)
        def assemble(p):
            return torch.cat((p[:2],p.new_zeros(d-5),p[2:]))
        class Small:
            def objective(self,p):return guide.objective(torch.stack([assemble(row) for row in p]))
        parameters,small_info=optimize(Small(),parameters,150)
        mean,full_info=optimize(guide,assemble(parameters),200)
        z=mean.detach().requires_grad_();value=guide.objective(z[None]).sum()
        gradient,=torch.autograd.grad(value,z)
        def conditional(p):
            # Replace only the five material/noise parameters, retaining the
            # optimized bridge-state reference in this derivative calculation.
            native=torch.cat((p[:2],mean[2:-3],p[2:]))
            return guide.objective(native[None]).sum()
        h=torch.autograd.functional.hessian(conditional,mean[indices],vectorize=True)
        h=(h+h.T)/2;eigen,vectors=torch.linalg.eigh(h)
        floor=max(.05,float(eigen.abs().max())*1e-7)
        metric=(vectors*eigen.clamp_min(floor))@vectors.T
        precision=torch.eye(d,dtype=mean.dtype)
        precision[indices[:,None],indices[None,:]]=metric
        decrement=float((gradient.detach()@torch.linalg.solve(precision,gradient.detach())).clamp_min(0).sqrt())
        record={'round':round_,'objective':float(value.detach()),'gradient_max_abs':float(gradient.abs().max()),
            'regularized_metric_decrement':decrement,'five_parameter_observed_eigenvalues':eigen.tolist(),
            'proposal_metric_floor':floor,'hyper_fit':small_info,'full_coordinate_fit':full_info}
        history.append(record)
        print('conditioned reference',record,flush=True)
        if checkpoint is not None:write(checkpoint/'conditioned_fit_progress.json',history)
        if round_==0:
            with torch.no_grad():anchor=guide.to_physical(mean[None])[0]
            parameters=mean[indices].detach().clone()
        else:break
    full_metric=None
    if hasattr(model,'floor_penalty'):
        precision,full_metric=full_curvature_reference(guide.objective,mean)
        decrement=float((gradient.detach()@torch.linalg.solve(precision,gradient.detach())).clamp_min(0).sqrt())
        history[-1]['full_observed_reference']=full_metric
        history[-1]['regularized_metric_decrement']=decrement
        print('full observed reference',full_metric,flush=True)
    stationary=decrement<.02 and float(gradient.abs().max())<.02
    if not stationary:raise ValueError('Conditioned-coordinate reference not stationary; sampling not started')
    return {'mean':mean,'precision':precision,'anchor':guide.anchor,'guide':guide.guide,
        'parameter_blocked_reference':hasattr(model,'floor_penalty'),
        'coordinates':'parameter-conditioned-gaussian-bridge-v1',
        'stationarity':{'stationary':True,'scope':'complete conditioned-coordinate density; not a physical-density MAP or a global optimum',
            'criterion':'max gradient < .02 and regularized metric decrement < .02',
            'regularized_metric_decrement':decrement,'history':history},
        'fit':{'seconds':time.time()-began,'initial_prior_reference':prior_info,
            'initial_prior_stationarity':prior_stationarity,
            'full_observed_reference':full_metric,
            'proposal':'parameter-dependent Gaussian state bridge with full nonlinear density/Jacobian correction; full observed fixed precision for floor-conditioned model, state I/5x5 parameter metric for historical model; two observed-only guide linearizations'}}
