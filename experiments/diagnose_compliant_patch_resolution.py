"""Bounded SECONDARY quadrature diagnosis; original manufactured run immutable.

Deliberately switches Gauss-Legendre to NESTED Clenshaw-Curtis5/9/17, without
changing any physical prior/truth/control/noise or the saved observation.
Every level is retained. No cohort/GPU/Isaac calls, adaptive node selection,
best-result selection or calibrated-physics claim.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,time,signal
from pathlib import Path
import numpy as np
import torch
try:
    import campaign_compliant_patch_inverse as original
except ImportError:
    from experiments import campaign_compliant_patch_inverse as original

ROOT=Path(__file__).resolve().parents[1]
ORDERS=(5,9,17)
WALL_CAP_S=55.
class DeadlineReached(Exception):pass


def cc_nodes_weights(order):
    if type(order) is not int:raise ValueError('Integer order required')
    n=order-1
    if n<4 or n&(n-1):raise ValueError('Nested power-of-two intervals required')
    angle=np.pi*np.arange(order)/n;x=np.cos(angle);x[n//2]=0.;x[0]=1.;x[-1]=-1.
    w=np.zeros(order);w[[0,-1]]=1/(n*n-1)
    v=np.ones(n-1)
    for k in range(1,n//2):v-=2*np.cos(2*k*angle[1:-1])/(4*k*k-1)
    v-=np.cos(n*angle[1:-1])/(n*n-1);w[1:-1]=2*v/n
    return x,w/2


def grid(order):
    nodes,w=cc_nodes_weights(order);rows=np.array(list(itertools.product(range(order),repeat=4)))
    theta=original.LOWER+(nodes[rows]+1)*(original.UPPER-original.LOWER)/2
    theta[:,[0,2,3]]=np.exp(theta[:,[0,2,3]])
    mass=np.prod(w[rows],axis=1)
    if (mass<=0).any() or abs(mass.sum()-1)>64*np.finfo(float).eps:raise ValueError('Normalized positive CC mass required')
    return theta,mass


def marginals(theta,weights):
    result=[]
    for j,name in enumerate(original.PARAMETERS):
        values,inverse=np.unique(theta[:,j],return_inverse=True)
        mass=np.bincount(inverse,weights=weights,minlength=len(values));cdf=np.cumsum(mass);cdf[-1]=1.
        q=values[np.searchsorted(cdf,[.05,.5,.95],side='left')]
        result.append(dict(name=name,values=values.tolist(),mass=mass.tolist(),cdf=cdf.tolist(),
            quantiles_05_50_95=q.tolist(),grid_nodes_in_90_percent_interval=int(np.sum((values>=q[0])&(values<=q[2])))))
    return result


def cdf_distance(first,second):
    knots=np.union1d(first['values'],second['values'])
    def evaluate(m):
        i=np.searchsorted(m['values'],knots,side='right')-1
        return np.where(i<0,0.,np.asarray(m['cdf'])[np.maximum(i,0)])
    return float(np.max(np.abs(evaluate(first)-evaluate(second))))


def compare(first,second):
    spans=np.array([8000.-2000.,.065-.005,.05-.018,30.-5.])
    parameters=[]
    for j,(a,b) in enumerate(zip(first['material_marginals'],second['material_marginals'])):
        ks=cdf_distance(a,b);shift=float(np.max(np.abs(np.array(a['quantiles_05_50_95'])-b['quantiles_05_50_95']))/spans[j])
        parameters.append(dict(name=a['name'],CDF_sup_distance=ks,maximum_quantile_shift_over_prior_span=shift,
            passed=ks<=.05 and shift<=.01))
    targets={}
    for label,start in [('A',25),('B',0)]:
        ma,sa=first['_moments'][label];mb,sb=second['_moments'][label]
        tolerance=max(2e-6,.02*float(np.sqrt(np.mean(sb[start:]**2))))
        dm=float(np.sqrt(np.mean((ma[start:]-mb[start:])**2)));ds=float(np.sqrt(np.mean((sa[start:]-sb[start:])**2)))
        targets[label]=dict(mean_RMS_change_m=dm,std_field_RMS_change_m=ds,tolerance_m=tolerance,passed=dm<=tolerance and ds<=tolerance)
    resolved=second['material_marginals'][0]['grid_nodes_in_90_percent_interval']>=3
    return dict(from_order=first['order'],to_order=second['order'],parameters=parameters,targets=targets,
        E_interval_spans_at_least3_nodes=resolved,passed=all(p['passed'] for p in parameters)
        and all(t['passed'] for t in targets.values()) and resolved)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def checked(r):
    p=ROOT/r['path']
    if sha(p)!=r['sha256']:raise ValueError('Pinned secondary input/source changed')
    return p
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def prepare(output):
    p=Path(output)
    if p.exists():raise ValueError('Fresh secondary protocol required')
    source=ROOT/'experiments/diagnose_compliant_patch_resolution.py'
    value=dict(schema='compliant-patch-secondary-resolution-v0.1',status='FROZEN_BEFORE_EXTRA_SOLVES',
        original_design=rec(ROOT/'runs/operator-dev/compliant-patch-inverse-v01/design.json'),
        original_summary=rec(ROOT/'runs/gaussian-campaign-v01/compliant-patch-inverse-v01/summary.json'),
        original_observation_archive=rec(ROOT/'runs/gaussian-campaign-v01/compliant-patch-inverse-v01/posterior-order5.npz'),
        source_sha256={source.relative_to(ROOT).as_posix():sha(source),
            **json.loads((ROOT/'runs/operator-dev/compliant-patch-inverse-v01/design.json').read_text())['source_sha256']},
        quadrature_family='Deliberate Gauss-Legendre to nested Clenshaw-Curtis numerical diagnosis',
        orders=list(ORDERS),particles_per_level=[o**4 for o in ORDERS],CPU_threads=2,chunk=512,
        wall_cap_s=WALL_CAP_S,hard_budget_behavior='Check before each bounded chunk; preserve all complete/partial levels and declare NOT qualified if timeout',
        criteria=dict(CDF_sup_distance=.05,quantile_shift_fraction_prior_physical_span=.01,
            predictive_field_RMS_absolute_floor_m=2e-6,predictive_field_RMS_fraction_posterior_std=.02,
            E_90_percent_interval_minimum_grid_nodes=3,
            overall='All3 levels complete; final adjacent comparison passes ALL parameterCDFs/quantiles, both targetmeans/stdfields, and material resolution guard'),
        ESS_interpretation='Quadrature weight concentration, NOT independent samples/MCMC mixing',
        no_physical_or_noise_or_control_retuning=True,same_saved_observation=True,
        no_filter_retry_or_best_selection=True,no_cohort_GPU_or_Isaac=True,
        scope='Numerical finite-grid diagnosis, never calibrated physics; original GL5 results preserved')
    p.parent.mkdir(parents=True,exist_ok=True);write(p,value);return rec(p)


def run(protocol_path,protocol_sha,output):
    protocol=json.loads(checked(dict(path=protocol_path,sha256=protocol_sha)).read_text())
    if protocol['orders']!=list(ORDERS) or protocol['wall_cap_s']!=WALL_CAP_S or protocol['chunk']!=512:raise ValueError('Fixed bounded plan required')
    for name,digest in protocol['source_sha256'].items():checked(dict(path=name,sha256=digest))
    old=json.loads(checked(protocol['original_summary']).read_text());checked(protocol['original_design'])
    with np.load(checked(protocol['original_observation_archive']),allow_pickle=False) as raw:observation=raw['observation'].copy()
    out=Path(output)
    if out.exists():raise ValueError('Fresh diagnosis output; preserve priorattempts')
    out.mkdir(parents=True);write(out/'protocol.json',protocol);torch.set_num_threads(2)
    started=time.monotonic();deadline=started+WALL_CAP_S;levels=[];comparisons=[];progress={}
    def clock():
        if time.monotonic()>=deadline:raise DeadlineReached('Declared55s cap reached; incomplete levels remain unqualified')
    def alarm(_signum,_frame):raise DeadlineReached('Declared55s wall alarm reached')
    old_handler=signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,WALL_CAP_S)
    try:
        truth={label:original.forward(original.TRUTH[None],label)[0][0] for label in ('A','B')}
        for order in ORDERS:
            theta,mass=grid(order);predicted=np.empty((len(theta),12));progress=dict(order=order,phase='likelihood',processed=0,planned=len(theta))
            for first in range(0,len(theta),512):
                clock();u,_=original.forward(theta[first:first+512],'A');predicted[first:first+len(u)]=original.selected_observations(u)
                progress['processed']=first+len(u)
            weights,info=original.condition(predicted,observation,mass)
            current=dict(order=order,particles=len(theta),inference=info,material_marginals=marginals(theta,weights),targets={},_moments={})
            np.savez_compressed(out/f'weights-order{order}.npz',theta=theta,prior_mass=mass,posterior_weights=weights)
            for label,start in [('A',25),('B',0)]:
                total=np.zeros((61,5,3));second=np.zeros_like(total);anchor=None
                progress=dict(order=order,phase='moments_'+label,processed=0,planned=len(theta))
                for first in range(0,len(theta),512):
                    clock();u,_=original.forward(theta[first:first+512],label)
                    if anchor is None:anchor=u[0].copy()
                    delta=u-anchor;w=weights[first:first+len(u)]
                    total+=np.einsum('b,btnc->tnc',w,delta);second+=np.einsum('b,btnc->tnc',w,delta**2)
                    progress['processed']=first+len(u)
                variance=second-total**2;bound=64*np.finfo(float).eps*(np.abs(second)+total**2+np.finfo(float).tiny)
                if np.any(variance < -bound):raise FloatingPointError('Centered moment precision failed')
                mean=anchor+total;std=np.sqrt(np.maximum(variance,0));current['_moments'][label]=(mean,std)
                current['targets'][label]=dict(mean_RMSE_m=float(np.sqrt(np.mean((mean[start:]-truth[label][start:])**2))),
                    latent_std_RMS_m=float(np.sqrt(np.mean(std[start:]**2))))
            if levels:comparisons.append(compare(levels[-1],current))
            levels.append(current)
            np.savez_compressed(out/f'moments-order{order}.npz',**{label+'_'+kind:value
                for label,pair in current['_moments'].items() for kind,value in zip(('mean','std'),pair)})
            write(out/f'level{order}.json',{k:v for k,v in current.items() if not k.startswith('_')})
        status='ALL_DECLARED_LEVELS_COMPLETED'
    except DeadlineReached:
        status='STOPPED_AT_DECLARED_WALL_CAP'
        write(out/'partial-level.json',dict(**progress,incomplete=True,not_qualified=True))
    except Exception as exc:
        write(out/'failure.json',dict(**progress,completed=False,error_type=type(exc).__name__,
            error=str(exc),elapsed_s=time.monotonic()-started,no_retry=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
    result=dict(schema='compliant-patch-secondary-resolution-result-v0.1',status=status,
        completed=status=='ALL_DECLARED_LEVELS_COMPLETED',protocol=dict(path=protocol_path,sha256=protocol_sha),
        original_GL5=old['levels'][-1],levels=[{k:v for k,v in x.items() if not k.startswith('_')} for x in levels],
        comparisons=comparisons,elapsed_s=time.monotonic()-started,
        declared_numerical_criterion_passed=status=='ALL_DECLARED_LEVELS_COMPLETED' and bool(comparisons) and comparisons[-1]['passed'],
        continuous_posterior_accuracy_proven=False,calibrated_physics_claim=False,raw_weights_moments_LOCAL=True)
    write(out/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('run')
    for name in ('protocol','protocol-sha256','output'):a.add_argument('--'+name,required=True)
    a=p.parse_args()
    value=prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)
    print(json.dumps(value if a.mode=='prepare' else {k:value[k] for k in ['status','elapsed_s','declared_numerical_criterion_passed']}))
