"""Separately planned numerical surrogate reference; originals remain immutable."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_k]='1'
import argparse,json,time,signal,multiprocessing
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr,ndtri,logsumexp
from scipy.stats import qmc
from scripts.compositional_neural_mechanics.conditional_inference_pilot import problem,predict,sha,write,OUT as OBSBASE
from scripts.compositional_neural_mechanics.joint_inference import neutral_parameters
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.reference_posterior_math import LOGK_MEAN,LOGK_BOUNDS,scale_bounds,compare
from scripts.compositional_neural_mechanics.reference_posterior_recovery_math import ConditionalCDF
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OLD=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-extension';OUT=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-surrogate';PLAN=R/'reference-posterior-surrogate-plan.json'

def observations():
    with np.load(OBSBASE/'observations-and-development-truth.npz')as a:
        return [{k:a[f'obs_{i}_{k}'].copy()for k in ('pose','points','displacement','forces','torques')}for i in range(2)]

def vector(pred,obs):
    return np.r_[np.concatenate([(p['displacement']-o['displacement']).ravel()/50e-6 for p,o in zip(pred,obs)]),np.concatenate([p[k].ravel()/s for p in pred for k,s in [('forces',.02),('torques',.001)]])]

def wrench_observed(obs):return np.concatenate([o[k].ravel()/s for o in obs for k,s in [('forces',.02),('torques',.001)]])

def design(x,degree):return np.stack([(x[:,0]/8)**i*(x[:,1]/8)**j for i in range(degree+1)for j in range(degree+1-i)],axis=1)

def train_polynomials():
    report=json.loads((R/'reference-posterior-extension-results.json').read_text());p=json.loads((OLD/'proposal.json').read_text());center=np.array(p['center']);L=np.linalg.cholesky(p['covariance']);xs=[];ys=[];obs=observations()
    for key in report['levels']['o25-std1']['node_keys']:
        row=json.loads((OLD/'nodes'/f'{key}.json').read_text());x=np.linalg.solve(L,np.array(row['ratio'])-center)
        if np.linalg.norm(x)>8:continue
        path=OLD/'nodes'/f'{key}.npz'
        if not row['completed'] or sha(path)!=row['response_sha256']:raise ValueError('Invalid training cache')
        with np.load(path)as a:pred=[{k:a[f'{i}_{k}'].copy()for k in ('displacement','forces','torques')}for i in range(2)]
        xs.append(x);ys.append(vector(pred,obs))
    x=np.array(xs);y=np.array(ys)
    if len(x)!=437:raise ValueError('Declared training census changed')
    return center,L,{d:np.linalg.lstsq(design(x,d),y,rcond=None)[0]for d in (3,4)}

def prepare():
    if PLAN.exists()or OUT.exists():raise ValueError('Preserve previous run')
    report=json.loads((R/'reference-posterior-extension-results.json').read_text());base=report['levels']['o33-std1']['posteriors']['VF']['2049'];p=json.loads((OLD/'proposal.json').read_text());rng=np.random.default_rng(927331)
    x=np.r_[rng.multivariate_normal(base['mean'],base['covariance'],16),rng.multivariate_normal(base['mean'],4*np.array(base['covariance']),16)];ratios=x[:,1:]-x[:,:1];angles=rng.uniform(0,2*np.pi,32);radius=rng.uniform(6,10,32);shell=np.stack([np.cos(angles),np.sin(angles)],1)*radius[:,None];ratios=np.r_[ratios,np.array(p['center'])+shell@np.linalg.cholesky(p['covariance']).T]
    sources=closure([Path(__file__)])+[ROOT/'tests/test_reference_posterior_surrogate.py'];inputs=[R/'reference-posterior-extension-results.json',R/'reference-posterior-extension-final-independent-audit.json',OLD/'proposal.json',OBSBASE/'observations-and-development-truth.npz']
    for k in report['levels']['o25-std1']['node_keys']:inputs.extend([OLD/'nodes'/f'{k}.json',OLD/'nodes'/f'{k}.npz'])
    plan=dict(scope='DEVELOPMENT three-parameter VF numerical surrogate reference, independently checked exact forward responses; not full15D posterior or rigorous global surrogate certificate',degrees=[3,4],primary_degree=4,training='Original order25 nodes, Euclidean radius<=8 under original proposal whitening;437nodes; fixed unweighted total-degree polynomial',fresh_validation_seed=927331,validation_ratios=ratios.tolist(),validation_groups=['proposal_width1']*16+['proposal_width2']*16+['radius6to10_shell']*32,validation_s=np.linspace(-1,1,9).tolist(),validation_extra='exact conditional mode clipped to support',gates=dict(position_vector_error=.01,wrench_vector_error=.01,position_component_error=.01,wrench_component_error=.01,absolute_loglikelihood_error=.02,mean_sd=.02,covariance_relative=.02,interval_sd=.05,log_evidence=.002,tail8_mass=1e-6,tail10_mass=1e-8,tailcut_interval_sd=.005),proposal_mean=base['mean'],proposal_covariance=base['covariance'],proposal_std_factors=[1,2],powers=[16,18,20],scramble_seeds=[927341,927342,927343,927344],integration='Scrambled Sobol3D Gaussian importance sampling, actual proposal density divided out; original independent bounded Gaussian priors; exact exp(s) wrench; empirical weighted quantiles, not Gaussian intervals',chunk=8192,workers=4,seconds=3600,node_seconds=120,tail_scope='Numerical radius8/10/full sensitivity and independent shell checks, not a mathematical global-tail bound',failure_policy='Any exact failure or validation gate failure prevents surrogate integration/qualification; all64 validation attempts retained',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in sources},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs})
    write(PLAN,plan)

def guard():
    p=json.loads(PLAN.read_text())
    for group in ('source_sha256','input_sha256'):
        for name,digest in p[group].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen source/input changed: '+name)
    return p

def exact_job(job):
    index,ratio,seconds=job;start=time.monotonic();row=dict(index=index,ratio=ratio,completed=False)
    def alarm(*args):raise TimeoutError('Exact-node timeout')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        torch.set_num_threads(1);obs=observations();model=problem([{k:v for k,v in o.items()if k not in ('forces','torques')}for o in obs],False);z=neutral_parameters();z[[0,3,5]]=[0.,*ratio];pred=[predict(model,z,o['pose'])for o in obs];v=vector(pred,obs)
        if not np.isfinite(v).all():raise ValueError('Nonfinite exact forward')
        path=OUT/'validation'/f'{index:03d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,response=v);row.update(completed=True,path=str(path.relative_to(ROOT)),sha256=sha(path))
    except Exception as e:row['error']=f'{type(e).__name__}: {e}'
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    row['seconds']=time.monotonic()-start;write(OUT/'validation'/f'{index:03d}.json',row);return row

def log_likelihood(v,s,y):
    return -.5*(np.sum(v[...,:156]**2,axis=-1)+np.sum((np.exp(s)[...,None]*v[...,156:]-y)**2,axis=-1))

def validation_identity(rows,plan):
    if len(rows)!=64 or sorted(r['index']for r in rows)!=list(range(64)):raise ValueError('Exact validation census must contain64 unique indices')
    for r in rows:
        if r['ratio']!=plan['validation_ratios'][r['index']]:raise ValueError('Exact validation ratio changed')
        if r['completed']:
            expected=OUT/'validation'/f"{r['index']:03d}.npz"
            if ROOT/r['path']!=expected or sha(expected)!=r['sha256']:raise ValueError('Exact validation artifact changed')

def validation_report(rows,plan,center,L,coefs,y):
    validation_identity(rows,plan)
    allrows=[]
    for row in rows:
        if not row['completed']:allrows.append(row);continue
        with np.load(ROOT/row['path'])as a:truth=a['response']
        ratio=np.array(row['ratio']);x=np.linalg.solve(L,ratio-center)[None,:];a=float(truth[156:]@truth[156:]);b=float(truth[156:]@y);c=float(y@y);conditional=ConditionalCDF(ratio,(a,b,c),'VF');s=np.array([z for z in plan['validation_s']if conditional.lo<=z<=conditional.hi]+[conditional.mode]);checks={}
        for d,coef in coefs.items():
            predicted=(design(x,d)@coef)[0];e=predicted-truth;checks[str(d)]=dict(position_vector_error=float(np.linalg.norm(e[:156])),wrench_vector_error=float(np.linalg.norm(e[156:])),position_component_error=float(np.abs(e[:156]).max()),wrench_component_error=float(np.abs(e[156:]).max()),absolute_loglikelihood_error=float(np.max(np.abs(log_likelihood(predicted,s,y)-log_likelihood(truth,s,y)))))
        passed=all(v<=plan['gates'][k]for c in checks.values()for k,v in c.items());allrows.append(dict(**row,group=plan['validation_groups'][row['index']],checks=checks,passed=passed))
    return dict(rows=allrows,passed=len(rows)==64 and all(r.get('passed',False)for r in allrows))

def summarize_samples(x,logw,radius):
    z=logsumexp(logw);w=np.exp(logw-z);mean=w@x;dx=x-mean;cov=(dx*w[:,None]).T@dx;sd=np.sqrt(np.diag(cov));interval=[]
    for j in range(3):
        order=np.argsort(x[:,j]);interval.append(np.interp([.05,.95],np.cumsum(w[order]),x[order,j]).tolist())
    return dict(log_evidence=float(z-np.log(len(x))),mean=mean.tolist(),covariance=cov.tolist(),sd=sd.tolist(),interval90=interval,importance_ESS=float(1/(w@w)),tail8_mass=float(w[radius>8].sum()),tail10_mass=float(w[radius>10].sum()),tail_sample_counts={str(t):int(np.sum(radius>t))for t in (8,10)},tail_effective_nodes={str(t):float(w[radius>t].sum()**2/np.sum(w[radius>t]**2))if np.sum(w[radius>t]**2)>0 else 0. for t in (8,10)})

def integration_level(plan,degree,coef,center,L,factor,seed,deadline):
    mean=np.array(plan['proposal_mean']);P=np.linalg.cholesky(plan['proposal_covariance'])*factor;u=qmc.Sobol(3,scramble=True,seed=seed).random_base2(max(plan['powers']));g=ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps));x=mean+g@P.T;lw=np.full(len(x),-np.inf);radius=np.zeros(len(x));y=wrench_observed(observations());normalizer=(ndtr(1)-ndtr(-1))*(ndtr(LOGK_BOUNDS[1]-LOGK_MEAN)-ndtr(LOGK_BOUNDS[0]-LOGK_MEAN))**2
    likelihood_constant=-156*np.log(50e-6*np.sqrt(2*np.pi))-12*np.log(.02*np.sqrt(2*np.pi))-12*np.log(.001*np.sqrt(2*np.pi))
    for start in range(0,len(x),plan['chunk']):
        if time.monotonic()>deadline:raise TimeoutError('Global integration budget')
        stop=min(start+plan['chunk'],len(x));xx=x[start:stop];r=xx[:,1:]-xx[:,:1];white=np.linalg.solve(L,(r-center).T).T;radius[start:stop]=np.linalg.norm(white,axis=1);v=design(white,degree)@coef
        logprior=-.5*np.sum((xx-[0,LOGK_MEAN,LOGK_MEAN])**2,axis=1)-1.5*np.log(2*np.pi)-np.log(normalizer);logq=-.5*np.sum(g[start:stop]**2,axis=1)-1.5*np.log(2*np.pi)-np.log(np.linalg.det(P));ll=log_likelihood(v,xx[:,0],y)+likelihood_constant;valid=(np.abs(xx[:,0])<=1)&(xx[:,1:]>=LOGK_BOUNDS[0]).all(1)&(xx[:,1:]<=LOGK_BOUNDS[1]).all(1);lw[start:stop]=np.where(valid,logprior+ll-logq,-np.inf)
    result={}
    for power in plan['powers']:
        n=2**power;row=summarize_samples(x[:n],lw[:n],radius[:n]);row.update(degree=degree,factor=factor,seed=seed,power=power)
        if power==max(plan['powers']):
            row['tailcut']={str(rad):summarize_samples(x[:n],np.where(radius[:n]<=rad,lw[:n],-np.inf),radius[:n])for rad in (8,10)}
        result[str(power)]=row
    return result

def run():
    plan=guard()
    if OUT.exists():raise ValueError('Preserve prior run')
    OUT.mkdir();write(OUT/'start.json',dict(plan_sha256=sha(PLAN),started_unix=time.time()));start=time.monotonic();deadline=start+plan['seconds'];center,L,coefs=train_polynomials();np.savez_compressed(OUT/'fixed-polynomials.npz',center=center,cholesky=L,degree3=coefs[3],degree4=coefs[4]);rows=[];result=dict(plan_sha256=sha(PLAN),complete=False,qualified=False,validation=None,integration={},checks={},failures=[])
    def save():result['elapsed_seconds']=time.monotonic()-start;write(R/'reference-posterior-surrogate-results.json',result)
    try:
        with multiprocessing.get_context('spawn').Pool(plan['workers'])as pool:
            jobs=[(i,r,plan['node_seconds'])for i,r in enumerate(plan['validation_ratios'])];it=pool.imap_unordered(exact_job,jobs)
            for _ in jobs:rows.append(it.next(timeout=max(.01,deadline-time.monotonic())));write(OUT/'validation-progress.json',rows)
        rows.sort(key=lambda x:x['index']);result['validation']=validation_report(rows,plan,center,L,coefs,wrench_observed(observations()));save()
        if not result['validation']['passed']:raise ValueError('Independent exact response validation failed; no integration')
        for degree in plan['degrees']:
            for factor in plan['proposal_std_factors']:
                for seed in plan['scramble_seeds']:
                    key=f'd{degree}-f{factor}-s{seed}';result['integration'][key]=integration_level(plan,degree,coefs[degree],center,L,factor,seed,deadline);save()
        high=str(max(plan['powers']));reference=result['integration'][f'd4-f1-s{plan["scramble_seeds"][0]}'][high]
        for key,levels in result['integration'].items():
            for power,row in levels.items():
                c=compare(row,reference);c['passed']=c['passed'] and c['log_evidence_change']<plan['gates']['log_evidence'];result['checks'][key+'-p'+power]=c
            row=levels[high];result['checks'][key+'-tail']=dict(passed=row['tail8_mass']<plan['gates']['tail8_mass']and row['tail10_mass']<plan['gates']['tail10_mass'],tail8_mass=row['tail8_mass'],tail10_mass=row['tail10_mass'])
            for radius,cut in row['tailcut'].items():
                c=compare(cut,row);c['passed']=c['passed']and c['interval_endpoint_change_in_sd']<plan['gates']['tailcut_interval_sd'];result['checks'][key+'-cut'+radius]=c
        old=json.loads((R/'reference-posterior-extension-results.json').read_text())['levels']['o33-std1']['posteriors']['VF']['2049'];c=compare(reference,old);c['passed']=c['mean_change_in_sd']<.02 and c['covariance_relative_change']<.02 and c['log_evidence_change']<plan['gates']['log_evidence'];c['note']='Only moments/evidence gated; original VF quantiles are unqualified';result['checks']['original_moments']=c;result['primary_summary']=reference;result['qualified']=all(c['passed']for c in result['checks'].values());result['complete']=True;guard()
    except Exception as e:
        result['qualified']=False
        result['failures'].append(f'{type(e).__name__}: {e}')
        if result['validation'] is None:
            seen={r['index']for r in rows}
            rows.extend(dict(index=i,ratio=r,completed=False,error='Global deadline or stage failure before terminal exact response')for i,r in enumerate(plan['validation_ratios'])if i not in seen)
            result['validation']=dict(passed=False,rows=sorted(rows,key=lambda r:r['index']))
    save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
