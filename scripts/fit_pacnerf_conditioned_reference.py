"""Fit an observed-data-only Gaussian-bridge reference, not a physical MAP."""
import math
import time
import torch
import pacnerf_innovation_metric as old_fitting
from pacnerf_discrepancy import DiscrepancyInnovation,gauss_newton
from pacnerf_conditioned_transport import ConditionedTransport
from pacnerf_comparison import optimize,write


def fit_reference(model,checkpoint=None,anchor=None):
    began=time.time();d=model.dimension
    if anchor is None:
        original=old_fitting.gauss_newton;old_fitting.gauss_newton=gauss_newton
        try:prior_mean,_,prior_stationarity,prior_info=old_fitting.fit_reference(DiscrepancyInnovation(model),checkpoint=checkpoint)
        finally:old_fitting.gauss_newton=original
        with torch.no_grad():anchor=DiscrepancyInnovation(model).to_physical(prior_mean[None])[0]
    else:prior_stationarity={'qualification':'provided observed-only development anchor'};prior_info={}
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
    stationary=decrement<.02 and float(gradient.abs().max())<.02
    if not stationary:raise ValueError('Conditioned-coordinate reference not stationary; sampling not started')
    return {'mean':mean,'precision':precision,'anchor':guide.anchor,'guide':guide.guide,
        'coordinates':'parameter-conditioned-gaussian-bridge-v1',
        'stationarity':{'stationary':True,'scope':'complete conditioned-coordinate density; not a physical-density MAP or a global optimum',
            'criterion':'max gradient < .02 and regularized metric decrement < .02',
            'regularized_metric_decrement':decrement,'history':history},
        'fit':{'seconds':time.time()-began,'initial_prior_reference':prior_info,
            'initial_prior_stationarity':prior_stationarity,
            'proposal':'parameter-dependent Gaussian state bridge, full nonlinear density/Jacobian correction; I on state coordinates and observed 5x5 material/noise metric; two observed-only guide linearizations'}}
