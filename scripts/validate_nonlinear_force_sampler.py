"""Independent quadrature check of nonlinear learned-scale force inference.

This validates the implemented sampler on a small independently integrable
posterior. It does not establish PAC renderer correctness or PAC calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from pacnerf_elliptical import sample
from physgauss.chain_diagnostics import diagnose_quantities
from pacnerf_comparison import write


OBS=np.array([-.55,.25,.9]);G=np.array([.6,-.2,.4]);V=np.array([1.,-.7,.3])


def quadrature(points):
    x=np.linspace(-5.,5.,points);h=np.linspace(-3.,3.,points)
    xx,hh=np.meshgrid(x,h,indexing='ij');sigma=.2*np.exp(hh)
    residual=OBS-(xx+xx**3/3)[...,None]*V
    inv=1/sigma**2;gg=G@G
    rg=residual@G;a=1+gg*inv
    quadratic=(residual**2).sum(-1)*inv-rg**2*inv**2/a
    logweight=-.5*xx**2-2*hh**2-.5*quadratic-3*np.log(sigma)-.5*np.log(a)
    # Product trapezoidal rule, calculated independently in NumPy.
    edge=np.ones(points);edge[[0,-1]]=.5
    weight=np.exp(logweight-logweight.max())*edge[:,None]*edge[None,:];weight/=weight.sum()
    amean=rg*inv/a;avar=1/a
    return {'x_mean':float((weight*xx).sum()),'h_mean':float((weight*hh).sum()),
        'a_mean':float((weight*amean).sum()),'x_second':float((weight*xx**2).sum()),
        'h_second':float((weight*hh**2).sum()),'a_second':float((weight*(avar+amean**2)).sum()),
        'x_grid':x,'x_cdf':np.cumsum(weight.sum(1)),'h_grid':h,'h_cdf':np.cumsum(weight.sum(0)),
        'posterior_boundary_mass':float(weight[[0,-1]].sum()+weight[:,[0,-1]].sum())}


def target(force):
    obs=torch.tensor(OBS,dtype=torch.float64);g=torch.tensor(G,dtype=torch.float64);v=torch.tensor(V,dtype=torch.float64)
    def evaluate(w):
        x=w[:,0];h=.5*w[:,1];sigma=.2*torch.exp(h)
        r=obs[None]-(x+x**3/3)[:,None]*v
        if force=='joint':
            r=r-w[:,2:3]*g
            nll=.5*r.square().sum(-1)/sigma.square()+3*torch.log(sigma)
        else:
            covariance=sigma[:,None,None].square()*torch.eye(3,dtype=w.dtype)+(g[:,None]*g[None,:])[None]
            root=torch.linalg.cholesky(covariance)
            whitened=torch.linalg.solve_triangular(root,r[...,None],upper=False)
            nll=.5*whitened.square().sum((1,2))+root.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        # White coordinates have exactly N(0,I) prior in both arms; it cancels
        # against the sampler's reference Gaussian, leaving the likelihood.
        return -nll,~torch.isfinite(nll)
    return evaluate


def validate(output,draws=12000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    coarse=quadrature(401);fine=quadrature(801)
    keys=['x_mean','h_mean','a_mean','x_second','h_second','a_second']
    grid_error=max(abs(coarse[k]-fine[k]) for k in keys)
    if grid_error>1e-5 or fine['posterior_boundary_mass']>1e-9:raise ValueError('Independent quadrature has not resolved the posterior')
    records=[]
    for index,force in enumerate(['marginal','joint']):
        dimension=2 if force=='marginal' else 3
        chains=sample(target(force),dimension,output/force,draws,warmup=1000,seed=660031+index,
            blocks=[(i,i+1) for i in range(dimension)],chunk=1000)
        w=chains['white'];x=w[:,:,0];h=.5*w[:,:,1]
        r=torch.tensor(OBS)-(x+x**3/3)[...,None]*torch.tensor(V)
        inv=(.2*torch.exp(h)).square().reciprocal();a=1+float(G@G)*inv
        amean=(r*torch.tensor(G)).sum(-1)*inv/a;asecond=1/a+amean.square()
        if force=='joint':amean=w[:,:,2];asecond=amean.square()
        quantities=torch.stack([x,h,amean,x.square(),h.square(),asecond],-1)
        diagnostic=diagnose_quantities(quantities,keys)
        errors={k:abs(float(quantities[:,:,i].mean())-fine[k]) for i,k in enumerate(keys)}
        cdf_errors={};quantile_errors={}
        for name,array in [('x',x),('h',h)]:
            values=np.sort(array.flatten().numpy());grid=fine[name+'_grid'];cdf=fine[name+'_cdf']
            # Quadrature CDF is a cell-center sum; midpoint interpolation
            # removes its O(dx) shift when comparing continuous distributions.
            cdf_mid=cdf-.5*np.diff(np.r_[0.,cdf])
            expected=np.interp(values,grid,cdf_mid)
            empirical=np.arange(1,len(values)+1)/len(values)
            cdf_errors[name]=float(np.max(np.abs(empirical-expected)))
            probs=np.array([.05,.5,.95]);actual=np.quantile(values,probs)
            predicted=np.interp(probs,cdf_mid,grid)
            quantile_errors[name]=float(np.max(np.abs(actual-predicted)))
        length=int(draws**.5);batches=draws//length
        batches_mean=quantities[:,:batches*length].reshape(4,batches,length,6).mean(2).flatten(0,1)
        mcse=batches_mean.std(0)/math.sqrt(len(batches_mean))
        moment_pass=all(errors[k]<max(.005,6*float(mcse[i])) for i,k in enumerate(keys))
        record={'force':force,'draws_per_chain':draws,'diagnostics':diagnostic,
            'moment_absolute_errors':errors,'moment_batch_mcse':dict(zip(keys,mcse.tolist())),
            'maximum_cdf_errors':cdf_errors,'maximum_5_50_95_quantile_errors':quantile_errors,
            'all_pass':diagnostic['all_monitored_quantities_pass'] and moment_pass and max(cdf_errors.values())<.025 and max(quantile_errors.values())<.05}
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='diagnostics'}),flush=True)
    result={'independent_reference':'401/801 point tensor-product quadrature; nonlinear observation x+x^3/3; unknown log-noise and Gaussian load',
        'seeds':{'marginal':660031,'joint':660032},'torch_version':torch.__version__,'numpy_version':np.__version__,
        'script_sha256_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'maximum_grid_moment_difference':grid_error,'posterior_boundary_mass':fine['posterior_boundary_mass'],
        'quadrature_moments':{k:fine[k] for k in keys},'arms':records,'all_pass':all(r['all_pass'] for r in records),
        'qualification':'Checks nonlinear posterior sampling, learned-scale normalizers and joint/marginal load consistency against quadrature. Not repeated-data PAC calibration or a guarantee of high-dimensional PAC exploration.'}
    write(output/'validation.json',result)
    if not result['all_pass']:raise ValueError('Nonlinear quadrature/sampler validation failed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--draws',type=int,default=12000)
    args=parser.parse_args();torch.set_num_threads(2);validate(args.output,args.draws)
