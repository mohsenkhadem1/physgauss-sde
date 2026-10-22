"""Prospective 64-noise fixed-truth coverage check; no Bayesian coverage promise."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
from scipy.stats import beta
from experiments import campaign_nonlinear_bar_inverse as bar
from experiments.campaign_bar_noise_population import PopulationPosterior

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NAMES=['experiments/campaign_bar_noise_population.py','experiments/run_bar_noise_population.py','tests/test_campaign_bar_noise_population.py']
SEED=2026121400
SUBSET=[0,9,18,27,36,45,54,63]
CAP=300

def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n');temp.replace(p)
def record(path):return {'path':str(Path(path).resolve().relative_to(ROOT)),'sha256':sha(path)}

def prepare(path):
    p=Path(path)
    if p.exists():raise ValueError('Fresh prospective design required before noise generation')
    old=ROOT/'runs/operator-dev/nonlinear-bar-inverse-v02/protocol.json'
    old_design=json.loads(old.read_text())
    sources={**old_design['source_sha256'],**{name:sha(ROOT/name) for name in SOURCE_NAMES}}
    design=dict(schema='nonlinear-bar-fixed-truth-noise-v01',status='FROZEN_BEFORE_NEW_NOISE',cases=64,noise_seed=SEED,
        independent_noise=True,old_single_case_excluded=True,truth_E_K=bar.TRUTH,prior='UNCHANGED loguniform E[4500,9000], K[100,400]',
        sigma_m=bar.SIGMA,known_forces_N=[1.,3.],held_force_N=2.,held_role='STATIC FORCE INTERPOLATION',CPU_threads=2,hard_wall_cap_s=CAP,
        algorithm=dict(conditional='c=1/K: prior1/c times Gaussian(mean=m(E),std=sigma/sqrt20); centered residualD(E) retains all4 raw channels',
            conditional_order=20,conditional_z_limit=10,maximum_uniform_inverse_power_series_and_tail_relative_bound=1e-14,
            whole_E_prior_and_ALL_noise_cases_required=True,no_clamping_redraw_or_case_drop=True,
            outer='Continuous adaptive1D quad_vec, maxnorm, limit256; per-case constant normalization shift only',
            outer_tolerances=[[1e-6,1e-12],[1e-8,1e-14]],quantile_probabilities=[.05,.95],quantile_bisections=23,
            moments='Centered and scaled, same fixed centers both passes; heuristic outer error estimates, not rigorous certificates'),
        independent_checks=dict(indices=SUBSET,algorithm='Direct2D GK21 cubature on ORIGINAL raw4-channel likelihood, loguniform unit coordinates',
            normalization_and_CDF_truth=['E','K','held-right'],centered_all4_moments=True,rtol=1e-8,atol=1e-14,max_subdivisions=512),
        numerical_gates=dict(maximum_CDF_difference=2e-4,maximum_parameter_quantile_unit_difference=1e-4,
            maximum_held_quantile_mean_std_difference_m=1e-7,maximum_parameter_mean_std_difference_in_prior_log_range=1e-5,
            maximum_independent_normalizer_relative_difference=5e-6,all64_coverage_classifications_identical=True,
            truth_CDF_cutoff_distance_minimum=2e-4,all64_and_fixed8_checks_required=True),
        coverage=dict(interval='equal-tailed latent90%',confidence_interval='Exact Clopper-Pearson95% for64 iid fixedtruth noise draws',
            quantities=['E','K','held-left','held-right'],metrics=['coverage','width','posterior-mean bias','RMSE'],
            no_forced90percent_pass_threshold=True,no_Bayesian_calibration_or_SBC_claim=True,
            near_edge='Report this truth\'s prior distances; no separate near-edge truth claim'),
        reference_protocol=record(old),source_sha256=sources,no_new_forward_physics_or_sampler=True,all_attempts_preserved=True)
    write(p,design);print(json.dumps(record(p)),flush=True)

def direct_checks(observations,fine,summary):
    obs=observations[SUBSET];shift=fine.shift[SUBSET];logs=bar.UPPER-bar.LOWER
    reports=[]
    def integrate(function,upper=np.ones(2)):
        result=cubature(function,np.zeros(2),np.asarray(upper),rule='gk21',rtol=1e-8,atol=1e-14,max_subdivisions=512)
        reports.append({'status':result.status,'subdivisions':result.subdivisions,'error':result.error})
        if result.status!='converged':raise RuntimeError('Independent2D cubature did not converge')
        return result.estimate
    def density(x):
        pred=bar.predict(bar.physical(x));delta=(pred[:,None,:]-obs[None])/bar.SIGMA
        return np.exp(shift[None]-.5*np.sum(delta**2,axis=2))
    Z=integrate(density);expected=fine.normalizer[SUBSET]/logs[1]
    cdf={}
    for j,key in enumerate(('E','K')):
        upper=np.ones(2);upper[j]=(np.log(bar.TRUTH[j])-bar.LOWER[j])/logs[j]
        cdf[key]=integrate(density,upper)/Z
    target=bar.predict(bar.TRUTH[None],[2.])[0,1]
    def right_region(x):
        E=np.exp(bar.LOWER[0]+logs[0]*x[:,0]);a=bar.SIZE[0]*(bar.stretch(E,[2.])[0][:,0]-1)
        denominator=target-a
        kmin=np.where(denominator>0,2/np.maximum(denominator,1e-100),np.inf)
        lower=np.clip((np.log(kmin)-bar.LOWER[1])/logs[1],0,1)
        transformed=x.copy();transformed[:,1]=lower+(1-lower)*x[:,1]
        return density(transformed)*(1-lower[:,None])
    cdf['right']=integrate(right_region)/Z
    center=np.array([6300.,200.,.01,.02]);scale=np.array([1000.,50.,.001,.001])
    def moment_integrand(x):
        theta=bar.physical(x);held=bar.predict(theta,[2.]);features=np.column_stack([theta,held]);d=(features-center)/scale
        w=density(x)
        return np.concatenate([w[:,:,None]*d[:,None,:],w[:,:,None]*d[:,None,:]**2],axis=2)
    moments=integrate(moment_integrand)/Z[:,None]
    means=center+scale*moments[:,:4];variance=moments[:,4:]-moments[:,:4]**2
    if np.min(variance)<0:raise RuntimeError('Independent centered variance negative')
    std=scale*np.sqrt(variance)
    maximum_cdf=max(float(np.max(abs(cdf[k]-summary[k]['cdf_truth'][SUBSET]))) for k in cdf)
    meanerr=np.stack([abs(means[:,j]-summary[k]['mean'][SUBSET]) for j,k in enumerate(('E','K','left','right'))],axis=1)
    stderr=np.stack([abs(std[:,j]-summary[k]['std'][SUBSET]) for j,k in enumerate(('E','K','left','right'))],axis=1)
    relative_Z=float(np.max(abs(Z/expected-1)))
    tolerance=np.array([4500*np.log(2)*1e-5,100*np.log(4)*1e-5,1e-7,1e-7])
    passed=maximum_cdf<=2e-4 and relative_Z<=5e-6 and np.all(meanerr<=tolerance) and np.all(stderr<=tolerance)
    return dict(indices=SUBSET,passed=bool(passed),maximum_CDF_difference=maximum_cdf,maximum_normalizer_relative_difference=relative_Z,
        maximum_mean_errors=meanerr.max(axis=0),maximum_std_errors=stderr.max(axis=0),CDFs=cdf,integration=reports)

def compare(a,b):
    rows={};passed=True
    for kind in a:
        x,y=a[kind],b[kind]
        qerr=max(float(np.max(abs(x[q]-y[q]))) for q in ('q05','q95'))
        if kind in ('E','K'):
            j=('E','K').index(kind)
            qerr=max(float(np.max(abs(np.log(x[q]/y[q]))/(bar.UPPER[j]-bar.LOWER[j]))) for q in ('q05','q95'))
            qtol=1e-4;mtol=(4500*np.log(2) if kind=='E' else 100*np.log(4))*1e-5
        else:qtol=mtol=1e-7
        cdferr=float(np.max(abs(x['cdf_truth']-y['cdf_truth'])))
        meanerr=float(np.max(abs(x['mean']-y['mean'])));stderr=float(np.max(abs(x['std']-y['std'])))
        covx=(x['cdf_truth']>=.05)&(x['cdf_truth']<=.95);covy=(y['cdf_truth']>=.05)&(y['cdf_truth']<=.95)
        distance=float(np.min(np.minimum(abs(y['cdf_truth']-.05),abs(y['cdf_truth']-.95))))
        good=cdferr<=2e-4 and qerr<=qtol and meanerr<=mtol and stderr<=mtol and np.array_equal(covx,covy) and distance>2e-4
        rows[kind]=dict(passed=bool(good),maximum_CDF_difference=cdferr,maximum_quantile_difference=qerr,
            maximum_mean_difference=meanerr,maximum_std_difference=stderr,coverage_classifications_equal=bool(np.array_equal(covx,covy)),
            minimum_truth_CDF_cutoff_distance=distance)
        passed=passed and good
    return dict(passed=bool(passed),quantities=rows)

def coverage(summary):
    truth=np.r_[bar.TRUTH,bar.predict(bar.TRUTH[None],[2.])[0]];rows={}
    for j,(kind,r) in enumerate(summary.items()):
        cover=(r['cdf_truth']>=.05)&(r['cdf_truth']<=.95);n=len(cover);k=int(cover.sum())
        ci=[0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k)]
        error=r['mean']-truth[j];width=r['q95']-r['q05']
        rows[kind]=dict(covered=k,total=n,coverage=k/n,exact_binomial_95percent_CI=ci,mean_width=float(width.mean()),median_width=float(np.median(width)),
            posterior_mean_bias=float(error.mean()),posterior_mean_RMSE=float(np.sqrt(np.mean(error**2))),truth=float(truth[j]))
    return rows

def run(protocol,digest,output):
    protocol=Path(protocol);output=Path(output)
    if sha(protocol)!=digest:raise ValueError('Protocol hash mismatch')
    design=json.loads(protocol.read_text())
    if design['status']!='FROZEN_BEFORE_NEW_NOISE' or design['noise_seed']!=SEED or design['cases']!=64:raise ValueError('Wrong prospective design')
    for name,wanted in design['source_sha256'].items():
        if sha(ROOT/name)!=wanted:raise ValueError('Frozen dependency hash mismatch: '+name)
    if output.exists():raise ValueError('Fresh output only; no retries')
    output.mkdir(parents=True);start=time.monotonic()
    state=dict(completed=False,numerically_qualified=False,pid=os.getpid(),protocol=record(protocol),raw_local=True,
        scope='Conditional frequentist coverage at ONE fixed truth, not SBC/populationphysics calibration',old_single_noise_case_excluded=True)
    def timeout(*_):raise TimeoutError('Prospective global300s wall cap reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CAP)
    try:
        exact=bar.independent_predict(bar.TRUTH[None])[0]
        observations=exact+np.random.default_rng(SEED).normal(0,bar.SIGMA,(64,4))
        np.savez(output/'observations.npz',observations_m=observations,exact_m=exact,seed=SEED,sigma_m=bar.SIGMA)
        state['observation_archive']=record(output/'observations.npz');write(output/'summary.json',state)
        summaries=[];posteriors=[]
        for rtol,atol in design['algorithm']['outer_tolerances']:
            print(f'Integrating all64 cases at rtol={rtol}',flush=True)
            p=PopulationPosterior(observations,rtol,atol);posteriors.append(p);result=p.summarize();summaries.append(result)
            state.setdefault('tolerance_passes',[]).append(dict(rtol=rtol,atol=atol,all64=result,series_bound=p.series_bound,
                heuristic_outer_error_max=max(p.errors),elapsed_s=time.monotonic()-start))
            write(output/'summary.json',state)
        state['tolerance_agreement']=compare(*summaries);write(output/'summary.json',state)
        print('Independent original2D checks on fixed8 indices',flush=True)
        state['independent_checks']=direct_checks(observations,posteriors[-1],summaries[-1]);write(output/'summary.json',state)
        state['coverage']=coverage(summaries[-1]);state['numerically_qualified']=state['tolerance_agreement']['passed'] and state['independent_checks']['passed']
        state['truth_logprior_coordinates']=(np.log(bar.TRUTH)-bar.LOWER)/(bar.UPPER-bar.LOWER)
        state['near_edge_truth_not_tested']=True;state['completed']=True
        if not state['numerically_qualified']:raise RuntimeError('All64 numerical qualification not met; coverage is provisional, no filtering')
    except BaseException as exc:
        state['error']=type(exc).__name__+': '+str(exc);state['traceback']=traceback.format_exc();raise
    finally:
        signal.alarm(0);state['elapsed_s']=time.monotonic()-start;state['terminal']=True
        write(output/'summary.json',state)
        print(json.dumps({k:state.get(k) for k in ('completed','numerically_qualified','elapsed_s','error')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare');parser.add_argument('--protocol');parser.add_argument('--sha256');parser.add_argument('--output');args=parser.parse_args()
    if args.prepare:prepare(args.prepare)
    else:run(args.protocol,args.sha256,args.output)
