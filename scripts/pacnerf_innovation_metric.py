"""Full-observation Gauss-Newton metric in exact causal innovation coordinates.

Only proposals use linearization. Optimization and every HMC evaluation use
all observed pixels and the exact nonlinear innovation map. No held-out data
are read. The metric is I + J_rgb.T J_rgb, assembled through the small poses.
"""
import math
import time
import torch


def pose_jacobian(innovation,mean,block_size=8):
    model=innovation.model
    frames=[f for f,c in model.specs if f]
    z=mean.detach().requires_grad_()
    physical=innovation.to_physical(z[None])
    q,_=model.decode(physical)
    values=q[0,frames].flatten()
    eye=torch.eye(len(values),dtype=z.dtype)
    result=z.new_empty(len(values),len(z))
    for start in range(0,len(values),block_size):
        stop=min(start+block_size,len(values))
        rows,=torch.autograd.grad(values,z,grad_outputs=eye[start:stop],
            is_grads_batched=True,retain_graph=stop<len(values))
        result[start:stop]=rows.detach()
    return q[0,frames].detach(),result


def gauss_newton(innovation,mean):
    started=time.time();model=innovation.model;m=model.modes
    poses,jac=pose_jacobian(innovation,mean)
    precision=torch.eye(len(mean),dtype=mean.dtype)
    observed=[spec for spec in model.specs if spec[0]]
    for slot,spec in enumerate(observed):
        def image(q):return (model.renderer(q[None,None],[spec])[0,0]/.08).flatten()
        image_jac=torch.autograd.functional.jacobian(image,poses[slot],
            vectorize=True,strategy='forward-mode').detach()
        local=image_jac@jac[slot*m:(slot+1)*m]
        precision+=local.T@local
    precision=(precision+precision.T)/2
    torch.linalg.cholesky(precision)
    return precision,{'metric':'I + full observed RGB Jacobian Gram in exact innovation coordinates',
        'minimum_eigenvalue':float(torch.linalg.eigvalsh(precision)[0]),'seconds':time.time()-started}


def fit_reference(innovation,checkpoint=None,rounds=60):
    from pacnerf_comparison import optimize,write
    mean,initial_fit=optimize(innovation,torch.zeros(innovation.dimension,dtype=torch.float64),350)
    history=[]
    for iteration in range(rounds):
        z=mean.detach().requires_grad_();value=innovation.objective(z[None]).sum()
        g,=torch.autograd.grad(value,z)
        precision,metric=gauss_newton(innovation,mean)
        step=torch.linalg.solve(precision,g.detach())
        decrement=float((step@g.detach()).clamp_min(0).sqrt())
        history.append({'iteration':iteration,'objective':float(value.detach()),
            'gradient_max_abs':float(g.abs().max()),'gauss_newton_decrement':decrement,'metric':metric})
        if checkpoint is not None:
            write(checkpoint/'innovation_fit_progress.json',history)
            torch.save({'mean':mean,'precision':precision},checkpoint/'innovation_fit_progress.pt')
        print('innovation fit',iteration,history[-1]['objective'],decrement,flush=True)
        if decrement<.02 and float(g.abs().max())<.02:
            return mean,precision,{'stationary':True,'criterion':'GN decrement < .02 and max gradient < .02; no claim of global optimum',
                'gauss_newton_decrement':decrement,'history':history},initial_fit
        transform=torch.linalg.solve_triangular(torch.linalg.cholesky(precision).T,
            torch.eye(len(mean),dtype=mean.dtype),upper=True)
        center=mean
        class White:
            def objective(self,w):return innovation.objective(center+w@transform.T)
        w,info=optimize(White(),torch.zeros_like(mean),150)
        candidate=center+w@transform.T
        with torch.no_grad():trial=float(innovation.objective(candidate[None])[0])
        if math.isfinite(trial) and trial<float(value.detach()):
            history[-1]['whitened_fit']=info;mean=candidate.detach();continue
        slope=float(g.detach()@step);fraction=1.
        for _ in range(24):
            candidate=mean-fraction*step
            try:
                with torch.no_grad():trial=float(innovation.objective(candidate[None])[0])
            except (ValueError,torch.linalg.LinAlgError):trial=math.inf
            if math.isfinite(trial) and trial<=float(value.detach())-1e-4*fraction*slope:
                mean=candidate.detach();break
            fraction*=.5
        else:raise ValueError('Innovation reference line search failed')
    raise ValueError('Innovation reference not stationary; sampling not started')
