"""Repeated-data uncertainty check including physical-feasibility conditioning.

Independent deterministic quadrature, not the PAC sampler. Rejection samples
the ENTIRE (scale,path) prior against the floor event; holding each scale fixed
while rejecting paths would generate a different model.
"""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from pacnerf_comparison import write


HEIGHT=.1;TOLERANCE=.02


def posterior(mean,sd,weights,grid,truth,hyper,zpoints=801):
    intervals=[];xranks=[];hranks=[]
    zg=torch.linspace(-9,9,zpoints,dtype=mean.dtype);dz=zg[1]-zg[0]
    normal=torch.exp(-.5*zg.square())/math.sqrt(2*math.pi)
    probabilities=mean.new_tensor([.25,.75,.1,.9,.05,.95,.025,.975])
    for start in range(0,len(mean),128):
        mu=mean[start:start+128];w=weights[start:start+128];n=len(mu)
        density=normal[None,None]*torch.sigmoid((HEIGHT+mu[:,:,None]+sd[None,:,None]*zg)/TOLERANCE)
        cumulative=torch.cumsum((density[:,:,1:]+density[:,:,:-1])*.5*dz,-1)
        cumulative=torch.cat((torch.zeros_like(cumulative[:,:,:1]),cumulative),-1)
        mass=cumulative[:,:,-1];total=(w*mass).sum(-1)
        def cdf(query):
            standardized=(query[:,:,None]-mu[:,None])/sd[None,None]
            index=((standardized-zg[0])/dz).floor().long().clamp(0,zpoints-2)
            fraction=((standardized-zg[index])/dz).clamp(0,1)
            table=cumulative[:,None].expand(-1,query.shape[1],-1,-1)
            lower=torch.gather(table,-1,index[...,None]).squeeze(-1)
            upper=torch.gather(table,-1,(index+1)[...,None]).squeeze(-1)
            return (w[:,None]*(lower+(upper-lower)*fraction)).sum(-1)/total[:,None]
        low=(mu-9*sd).amin(-1)[:,None].expand(-1,8).clone()
        high=(mu+9*sd).amax(-1)[:,None].expand(-1,8).clone()
        for _ in range(44):
            middle=(low+high)/2;value=cdf(middle)
            low=torch.where(value<probabilities,middle,low);high=torch.where(value>=probabilities,middle,high)
        intervals.append((low+high)/2);xranks.append(cdf(truth[start:start+n,None])[:,0])
        # P(h|y,F) includes the integrated missing-position feasibility mass.
        wh=w*mass;wh/=wh.sum(-1,keepdim=True)
        ch=wh.cumsum(-1)-.5*wh
        index=torch.searchsorted(grid,hyper[start:start+n]).clamp(1,len(grid)-1)
        fraction=((hyper[start:start+n]-grid[index-1])/(grid[index]-grid[index-1])).clamp(0,1)
        hranks.append(ch.gather(1,(index-1)[:,None])[:,0]*(1-fraction)+ch.gather(1,index[:,None])[:,0]*fraction)
    return torch.cat(intervals),torch.cat(xranks),torch.cat(hranks)


def ks_uniform(ranks):
    ordered=ranks.sort().values;ecdf=torch.arange(1,len(ranks)+1,dtype=ranks.dtype)/len(ranks)
    return float(torch.maximum((ecdf-ordered).abs(),(ecdf-1/len(ranks)-ordered).abs()).max())


def validate(output,datasets=2000):
    dtype=torch.float64;rng=torch.Generator().manual_seed(58211)
    time=torch.tensor([.25,.5,1.],dtype=dtype)
    low=torch.minimum(time[:,None],time[None]);high=torch.maximum(time[:,None],time[None])
    process=.4**2*low.square()*(3*high-low)/6
    force=.8*(time.square()/2)[:,None];accepted=[];hypers=[];proposed=0
    while sum(len(x) for x in accepted)<datasets:
        h=torch.randn(512,dtype=dtype,generator=rng)
        covariance=process[None]*torch.exp(2*h)[:,None,None]+force@force.T
        x=(torch.linalg.cholesky(covariance)@torch.randn(512,3,1,dtype=dtype,generator=rng)).squeeze(-1)
        logevent=-torch.nn.functional.softplus(-(HEIGHT+x)/TOLERANCE).sum(-1)
        keep=torch.log(torch.rand(512,dtype=dtype,generator=rng))<logevent
        accepted.append(x[keep]);hypers.append(h[keep]);proposed+=512
    truth=torch.cat(accepted)[:datasets];hyper=torch.cat(hypers)[:datasets]
    observed=torch.tensor([0,2]);y=truth[:,observed]
    grid=torch.linspace(-6,6,121,dtype=dtype)
    prior=process[None]*torch.exp(2*grid)[:,None,None]+force@force.T
    yy=prior[:,observed][:,:,observed];cross=prior[:,1,observed]
    chol=torch.linalg.cholesky(yy);gain=torch.cholesky_solve(cross[...,None],chol).squeeze(-1)
    variance=prior[:,1,1]-(gain*cross).sum(-1);sd=variance.sqrt()
    mean=y@gain.T
    whitened=torch.linalg.solve_triangular(chol[None],y[:,None,:,None],upper=False).squeeze(-1)
    logweight=-.5*grid.square()[None]-.5*whitened.square().sum(-1)-chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)[None]
    weights=logweight.softmax(-1)
    quantiles,xr,hr=posterior(mean,sd,weights,grid,truth[:,1],hyper)
    fine,_,_=posterior(mean[:64],sd,weights[:64],grid,truth[:64,1],hyper[:64],1601)
    quadrature_error=float((fine-quantiles[:64]).abs().max())
    records=[]
    for i,level in enumerate([.5,.8,.9,.95]):
        included=(truth[:,1]>=quantiles[:,2*i])&(truth[:,1]<=quantiles[:,2*i+1])
        coverage=float(included.double().mean());se=math.sqrt(level*(1-level)/datasets)
        records.append({'nominal':level,'coverage':coverage,'nominal_binomial_mc_standard_error':se,
            'within_four_mc_standard_errors':abs(coverage-level)<4*se,
            'mean_width_m':float((quantiles[:,2*i+1]-quantiles[:,2*i]).mean())})
    bound=math.sqrt(math.log(200)/(2*datasets));xks=ks_uniform(xr);hks=ks_uniform(hr)
    result={'datasets':datasets,'seed':58211,'proposed_prior_datasets':proposed,
        'generative_model':'Integrated Gaussian acceleration plus Gaussian global force; h~N(0,1) BEFORE jointly conditioning (h,path) on F=1.',
        'floor_event':'P(F=1|path)=product sigmoid((.1+displacement)/.02) at all3 times',
        'observations':'Exact positions at t=.25 and1; missing t=.5; observed-position event factors cancel only after conditioning on their values.',
        'posterior':'Independent121-point scale and801-point conditional-standard-normal quadrature, including missing-position feasibility mass in scale weights; mixture CDF inversion.',
        'coverage':records,'position_pit_ks':xks,'hyperparameter_sbc_ks':hks,'dkw_99_percent_bound':bound,
        '801_vs_1601_maximum_quantile_difference_first64_m':quadrature_error,
        'conditional_prior_log_scale_mean':float(hyper.mean()),
        'all_pass':all(r['within_four_mc_standard_errors'] for r in records) and max(xks,hks)<bound and quadrature_error<.001,
        'script_sha256_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'qualification':'Same-model small linear feasibility/force/scale uncertainty check; not nonlinear PAC posterior convergence or independent real-data calibration.'}
    write(Path(output)/'validation.json',result);print(json.dumps(result,indent=2),flush=True)
    if not result['all_pass']:raise ValueError('Floor-conditioned uncertainty validation failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--datasets',type=int,default=2000)
    a=p.parse_args();torch.set_num_threads(2);validate(a.output,a.datasets)
