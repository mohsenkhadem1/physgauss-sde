"""Independent repeated-data validation of Gaussian-force/scale marginalization.

This small linear integrated-acceleration experiment permits independent grid
integration over a scale posterior. It validates mixture interval arithmetic
and learned-scale normalizers, not the nonlinear PAC sampler or PAC calibration.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from pacnerf_comparison import write


def validate(output,datasets=2000,seed=55201):
    rng=torch.Generator().manual_seed(seed);dtype=torch.float64
    time=torch.arange(1,9,dtype=dtype)/8
    lo=torch.minimum(time[:,None],time[None]);hi=torch.maximum(time[:,None],time[None])
    process=.4**2*lo.square()*(3*hi-lo)/6
    force=.8*(time.square()/2)[:,None]
    observed=torch.tensor([0,2,7]);missing=torch.tensor([3,4,5,6]);sensor=.03
    hyper=torch.randn(datasets,dtype=dtype,generator=rng)
    covariance=process[None]*torch.exp(2*hyper)[:,None,None]+force@force.T
    truth=(torch.linalg.cholesky(covariance)@torch.randn(datasets,8,1,dtype=dtype,generator=rng)).squeeze(-1)
    y=truth[:,observed]+sensor*torch.randn(datasets,3,dtype=dtype,generator=rng)
    grid=torch.linspace(-6,6,121,dtype=dtype)
    prior=process[None]*torch.exp(2*grid)[:,None,None]+force@force.T
    yy=prior[:,observed][:,:,observed]+sensor**2*torch.eye(3,dtype=dtype)
    cross=prior[:,missing][:,:,observed]
    chol=torch.linalg.cholesky(yy)
    gain=torch.cholesky_solve(cross.transpose(-1,-2),chol).transpose(-1,-2)
    conditional=prior[:,missing][:,:,missing]-gain@cross.transpose(-1,-2)
    variance=conditional.diagonal(dim1=-2,dim2=-1)
    whitened=torch.linalg.solve_triangular(chol[None],y[:,None,:,None],upper=False).squeeze(-1)
    logweight=-.5*grid.square()[None]-.5*whitened.square().sum(-1)-chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)[None]
    weights=logweight.softmax(-1)
    mean=torch.einsum('gmo,bo->bgm',gain,y)
    sigma=variance.sqrt()[None]
    # Actual mixture quantiles, not mean +/- normal quantile times total SD.
    low=(mean-12*sigma).amin(1);high=(mean+12*sigma).amax(1)
    def quantile(probability):
        left,right=low.clone(),high.clone()
        for _ in range(48):
            middle=(left+right)/2
            cdf=(weights[...,None]*torch.special.ndtr((middle[:,None]-mean)/sigma)).sum(1)
            left=torch.where(cdf<probability,middle,left);right=torch.where(cdf>=probability,middle,right)
        return (left+right)/2
    records=[]
    for level in [.5,.8,.9,.95]:
        lower,upper=quantile((1-level)/2),quantile((1+level)/2)
        included=(truth[:,missing]>=lower)&(truth[:,missing]<=upper)
        coverage=included.double().mean(0)
        # Per-coordinate datasets are independent; coordinates within one draw
        # are correlated and are never counted as extra independent datasets.
        se=math.sqrt(level*(1-level)/datasets)
        records.append({'nominal':level,'coverage_each_gap_time':coverage.tolist(),
            'nominal_binomial_mc_standard_error':se,'mean_width_m':float((upper-lower).mean()),
            'within_four_mc_standard_errors':bool((torch.abs(coverage-level)<4*se).all())})
    # Independent total-covariance identity for the mixture and hyper SBC CDF.
    mixture_mean=(weights[...,None]*mean).sum(1)
    within=(weights[...,None]*variance[None]).sum(1)
    between=(weights[...,None]*(mean-mixture_mean[:,None]).square()).sum(1)
    second=(weights[...,None]*(variance[None]+mean.square())).sum(1)-mixture_mean.square()
    error=float((second-within-between).abs().max())
    # Smoothly interpolate a trapezoidal posterior density for SBC CDFs.
    density=weights/(grid[1]-grid[0]);cdf=torch.cumsum((density[:,1:]+density[:,:-1])*.5*(grid[1]-grid[0]),1)
    cdf=torch.cat((torch.zeros(datasets,1,dtype=dtype),cdf),1);cdf=cdf/cdf[:,-1:]
    index=torch.searchsorted(grid,hyper).clamp(1,len(grid)-1)
    fraction=(hyper-grid[index-1])/(grid[index]-grid[index-1])
    ranks=cdf.gather(1,(index-1)[:,None]).squeeze(1)*(1-fraction)+cdf.gather(1,index[:,None]).squeeze(1)*fraction
    ranks=ranks.clamp(0,1)
    sorted_ranks=ranks.sort().values;empirical=torch.arange(1,datasets+1,dtype=dtype)/datasets
    ks=float(torch.maximum((empirical-sorted_ranks).abs(),(empirical-1/datasets-sorted_ranks).abs()).max())
    result={'datasets':datasets,'seed':seed,'independent_datasets':True,'observed_time_indices':observed.tolist(),
        'gap_time_indices':missing.tolist(),'unknown_log_diffusion_prior':'N(0,1)',
        'posterior_computation':'121-point deterministic quadrature on [-6,6]; includes Gaussian data determinant; mixture-CDF bisection',
        'coverage':records,'hyperparameter_sbc_cdf_histogram_10_bins':torch.histc(ranks,bins=10,min=0,max=1).long().tolist(),
        'hyperparameter_sbc_ks':ks,'dkw_99_percent_bound':math.sqrt(math.log(200)/(2*datasets)),
        'maximum_total_covariance_identity_error':error,
        'all_pass':all(r['within_four_mc_standard_errors'] for r in records) and error<1e-10 and ks<math.sqrt(math.log(200)/(2*datasets)),
        'qualification':'independent linear-Gaussian hierarchical mixture arithmetic check only; does not validate PAC dynamics, nonlinear HMC exploration, transferred neural diffusion or real-world calibration'}
    write(Path(output)/'validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['all_pass']:raise ValueError('Repeated-data hierarchical validation failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--datasets',type=int,default=2000)
    a=p.parse_args();torch.set_num_threads(2);validate(a.output,a.datasets)
