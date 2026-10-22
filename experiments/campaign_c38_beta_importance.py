"""Fixed exact-likelihood correction of frozen neural proposals, DEV/CAL only.

Scrambled-QMC weighted clouds are numerical integration, not MCMC or iid
posterior samples. The learned density is a PROPOSAL, never a changed prior.
All acquired observations enter the exact likelihood once. No TEST access.
"""
from __future__ import annotations
import argparse,json,os,resource,signal,time
from pathlib import Path
import numpy as np
from scipy.special import betaincinv,logsumexp
from scipy.stats import qmc
from experiments import run_campaign_c38_beta_control as frozen
from experiments.campaign_c38_beta_metrics import log_density_numpy
from experiments.campaign_c38_beta_posterior import numpy_parameters
from experiments.campaign_c38_bar_posterior_z import BarPosteriorBatch
from experiments.campaign_c38_bar_reference import LOWER,UPPER,SIGMA_M,TARGET_FORCE,prior_physical,endpoint_displacements,oracle_positions,material_quadrature

ROOT=Path(__file__).resolve().parents[1]
PARENT_PROTOCOL='runs/operator-dev/c38-beta-control-v01/protocol-v01.json'
PARENT_SHA='b0a92c1825e0f995b81fa3e9c6fe0267681771c45b8e324d83563804279f4092'
BUDGETS=(1024,4096,16384,65536);SCRAMBLES=(2026122450,2026122451,2026122452,2026122453)
CRITERIA={'CDF_grid':.005,'field_mean_m':2e-6,'field_std_m':2e-6,'relative_evidence':.005,'ESS_min':400.,'maximum_weight':.01,'between_scramble_CDF_grid':.005,'between_scramble_field_mean_m':2e-6,'between_scramble_field_std_m':2e-6}
SOURCE='experiments/campaign_c38_beta_importance.py'

def sobol_points(seed,power=16):
    u=qmc.Sobol(d=4,scramble=True,seed=seed).random_base2(power)
    if not np.isfinite(u).all() or ((u<=0)|(u>=1)).any():raise FloatingPointError('Finite Sobol endpoint encountered; no clipping/redraw')
    return u

def sample_proposal(uniforms,parameters=None):
    """Exactly.5 prior+.5 normalized NN using4D scrambled Sobol coordinates."""
    u=np.asarray(uniforms,dtype=float)
    if u.ndim!=2 or u.shape[1]!=4 or not np.isfinite(u).all() or ((u<=0)|(u>=1)).any():raise ValueError('Strictly interior4D uniforms required')
    unit=u[:,2:].copy()
    if parameters is not None:
        p=numpy_parameters(parameters)
        if len(p['weights'])!=1:raise ValueError('Exactly one conditional neural proposal required')
        branch=u[:,0]>=.5;ids=np.searchsorted(np.cumsum(p['weights'][0]),u[branch,1]);
        if (ids>=len(p['weights'][0])).any():raise FloatingPointError('Normalized component CDF lost support')
        unit[branch]=betaincinv(p['alpha'][0,ids],p['beta'][0,ids],u[branch,2:])
    if not np.isfinite(unit).all() or ((unit<=0)|(unit>=1)).any():raise FloatingPointError('Beta inverse-CDF lost interior support; preserve failure, no clipping/redraw')
    return unit

def exact_log_likelihood(unit,observed,forces,sides):
    y=np.asarray(observed,dtype=float);F=np.asarray(forces,dtype=float);side=np.asarray(sides)
    if y.shape!=F.shape or F.ndim!=1 or side.shape!=F.shape or not np.isfinite(y).all() or not np.isin(side,[0,1]).all():raise ValueError('Exact acquired observation/control tuple required')
    prediction=endpoint_displacements(prior_physical(unit),F)[:,np.arange(len(F)),side.astype(int)]
    # Omitting the known measurement-density constant in BOTH normalizers.
    return -.5*np.square((prediction-y)/SIGMA_M).sum(1)

def proposal_log_density(unit,parameters=None):
    if parameters is None:return np.zeros(len(unit))
    return np.logaddexp(-np.log(2.),-np.log(2.)+log_density_numpy(unit,parameters))

def evaluate_cloud(unit,observed,forces,sides,parameters=None,*,chunk=8192):
    X=material_quadrature();logw=[];displacement=[]
    for start in range(0,len(unit),chunk):
        u=unit[start:start+chunk];logw.append(exact_log_likelihood(u,observed,forces,sides)-proposal_log_density(u,parameters));displacement.append((oracle_positions(prior_physical(u),TARGET_FORCE,X)-X).reshape(len(u),-1))
    return np.concatenate(logw),np.concatenate(displacement)

def summarize_cloud(unit,logw,displacement,grid):
    u=np.asarray(unit,dtype=float);lw=np.asarray(logw,dtype=float);q=np.asarray(displacement,dtype=float);N=len(u)
    if u.shape!=(N,2) or lw.shape!=(N,) or q.shape!=(N,24) or not all(np.isfinite(v).all() for v in (u,lw,q)):raise ValueError('Finite complete weighted material cloud required')
    logsum=logsumexp(lw);w=np.exp(lw-logsum);mean=w@q;std=np.sqrt(w@((q-mean)**2));cdf=np.array([[w[u[:,j]<=x].sum() for x in grid] for j in range(2)]);quant=[]
    for j in range(2):
        order=np.argsort(u[:,j]);cw=np.cumsum(w[order]);indices=np.searchsorted(cw,[.05,.5,.95]);quant.append(u[order[np.minimum(indices,N-1)],j])
    return dict(count=N,log_evidence=float(logsum-np.log(N)),CDF_grid=cdf,field_mean=material_quadrature()+mean.reshape(8,3),field_std=std.reshape(8,3),quantiles_unit=np.asarray(quant),ESS=float(1/(w@w)),maximum_weight=float(w.max()),largest1percent_mass=float(np.sort(w)[-max(1,int(np.ceil(.01*N))):].sum()),finite_prior_tail_mass_by_latent=np.array([[w[u[:,j]<=.05].sum(),w[u[:,j]>=.95].sum()] for j in range(2)]),weighted_QMC_cloud_not_MCMC_or_iid_samples=True)

def compare(summary,reference):
    error=dict(CDF_grid=float(np.max(abs(summary['CDF_grid']-reference['marginal_CDF_grid']))),field_mean_m=float(np.max(abs(summary['field_mean']-reference['field_mean']))),field_std_m=float(np.max(abs(summary['field_std']-reference['field_std']))),relative_evidence=float(abs(np.expm1(summary['log_evidence']-reference['log_evidence']))))
    passed=all(v<=CRITERIA[k] for k,v in error.items()) and summary['ESS']>=CRITERIA['ESS_min'] and summary['maximum_weight']<=CRITERIA['maximum_weight']
    return dict(errors=error,passed=bool(passed))

def source_names():return [SOURCE]+frozen.SOURCES

def prepare(path,output):
    if Path(path).exists() or Path(output).exists():raise FileExistsError('Fresh protocol/output only')
    parent=frozen.validate(ROOT/PARENT_PROTOCOL,PARENT_SHA);out=ROOT/parent['output'];validation=out/'validation/summary.json';v=json.loads(validation.read_text());dataset=json.loads((out/'dataset.json').read_text())
    if v.get('completed') is not True or v.get('all_gates_passed') is not False or v.get('protocol')!=frozen.record(ROOT/PARENT_PROTOCOL):raise ValueError('Completed frozen original neural negative report required')
    if (out/'validation/test-release.json').exists():raise ValueError('TEST must remain closed')
    inputs={'parent_protocol':frozen.record(ROOT/PARENT_PROTOCOL),'validation':frozen.record(validation),'fit':frozen.record(out/'fits/execution.json'),'dataset':frozen.record(out/'dataset.json')};roles={}
    for role in ('DEV','CAL'):
        roles[role]=dict(public=dataset['roles'][role]['public'],parameters={str(s):frozen.record(out/f'validation/{role}-seed-{s}-parameters.npz') for s in frozen.FIT_SEEDS},references={str(c):frozen.record(out/f'validation/{role}-reference-context-{c}.json') for c in range(8)})
    p=dict(schema='c38-beta-fixed-importance-v01',status='FROZEN_BEFORE_NEW_CORRECTION_EVALUATIONS',output=frozen.rel(output),source_sha256={n:frozen.sha(ROOT/n) for n in source_names()},inputs=inputs,roles=roles,object_indices=list(range(16)),context_assignment='object_index modulo8; exactly2/context/role',budgets=BUDGETS,scramble_seeds=SCRAMBLES,fit_seeds=frozen.FIT_SEEDS,proposal='0.5 uniform unit(logE,logK) prior +0.5 FROZEN neural density conditional on ALL acquired observations; proposal only, not a changed prior',baseline='Uniform prior, equal hypothesis counts; same4D scrambled Sobol u2/u3; reused for comparisons to all3nets, not3 independent baseline runs',weight='exp(raw all-acquired Gaussian log-likelihood)/proposal_density; intrinsic unit prior density1, no double-counting or extra logK Jacobian',nested_prefixes=True,all_budgets_retained_no_early_success_selection=True,comparison=CRITERIA,grid=parent['validation']['CDF_grid'],resources=parent['resources'],cpu_threads=2,wall_cap_seconds=900,chunk=8192,reference=dict(existing='Hash-bound continuous CDF/field reports; no invented normalizer from CDF agreement',new='Instantiate original qualified low/high BarPosteriorBatch on SAME DEV/CAL acquired measurements ONLY to obtain log evidence; record added cost',lower=parent['validation']['reference_lower'],higher=parent['validation']['reference_higher'],relative_normalizer_tolerance=frozen.GATES['reference_relative_Z']),guards=dict(TEST_never_opened=True,no_fit_or_hyperparameter_changes=True,no_truth_parameter_or_canonical_target_labels=True,no_endpoint_clipping_redraw=True),scope='Numerical corrected posterior for32 analytical DEV/CAL cases, not neural/population/realphysics calibration; finitegrid/functionals and4scrambles do not prove fulljoint convergence',raw_local=True)
    Path(path).parent.mkdir(parents=True,exist_ok=True);frozen.write(path,p);return frozen.sha(path)

def validate(path,digest):
    if frozen.sha(path)!=digest:raise ValueError('Corrector protocol changed')
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='c38-beta-fixed-importance-v01' or p.get('budgets')!=list(BUDGETS) or p.get('scramble_seeds')!=list(SCRAMBLES) or p.get('fit_seeds')!=list(frozen.FIT_SEEDS) or p.get('object_indices')!=list(range(16)) or p.get('comparison')!=CRITERIA or set(p.get('roles',{}))!={'DEV','CAL'} or set(p.get('source_sha256',{}))!=set(source_names()):raise ValueError('Exact prospective numerical study required')
    for n,d in p['source_sha256'].items():
        if frozen.sha(ROOT/n)!=d:raise ValueError('Corrector source changed: '+n)
    for rec in p['inputs'].values():frozen.checked(rec)
    for role,r in p['roles'].items():
        if set(r['parameters'])!=set(map(str,frozen.FIT_SEEDS)) or set(r['references'])!=set(map(str,range(8))):raise ValueError('All fixed proposal seeds/reference contexts required')
        frozen.checked(r['public'])
        for rec in list(r['parameters'].values())+list(r['references'].values()):frozen.checked(rec)
    frozen.threads();frozen.guard_resources(p);return p

def reference_group(p,public,role,c):
    ids=np.array([i for i in p['object_indices'] if i%8==c]);mask=public['mask'][0,c];F=public['forces'][0,c,mask];sides=public['sides'][0,c,mask].astype(int);y=public['observed_m'][ids,c][:,mask];saved=frozen.load_record(p['roles'][role]['references'][str(c)])
    if saved.get('passed') is not True:raise ValueError('Continuous reference report did not qualify')
    started=time.monotonic();low=BarPosteriorBatch(y,F,sides,**p['reference']['lower']);high=BarPosteriorBatch(y,F,sides,**p['reference']['higher']);lowlog=np.log(low.Z)-low.shift+np.log(low.std*np.sqrt(2*np.pi)/(UPPER[1]-LOWER[1]));highlog=np.log(high.Z)-high.shift+np.log(high.std*np.sqrt(2*np.pi)/(UPPER[1]-LOWER[1]));error=np.max(abs(np.expm1(lowlog-highlog)))
    references=[{k:np.asarray(saved[k])[j] for k in ('marginal_CDF_grid','field_mean','field_std')} for j in range(len(ids))]
    for j,r in enumerate(references):r.update(log_evidence=float(highlog[j]),new_reference_normalizer_relative_error=float(error),new_reference_group_seconds=time.monotonic()-started)
    return ids,y,F,sides,references

def final_case_summary(records):
    high=[r['budgets'][-1] for r in records];spread=dict(between_scramble_CDF_grid=float(np.ptp(np.asarray([r['moments']['CDF_grid'] for r in high]),axis=0).max()),between_scramble_field_mean_m=float(np.ptp(np.asarray([r['moments']['field_mean'] for r in high]),axis=0).max()),between_scramble_field_std_m=float(np.ptp(np.asarray([r['moments']['field_std'] for r in high]),axis=0).max()))
    return dict(passed=all(r['comparison']['passed'] for r in high) and all(v<=CRITERIA[k] for k,v in spread.items()),spread=spread,scrambles=len(records),largest_budget=BUDGETS[-1],worst_errors={k:max(r['comparison']['errors'][k] for r in high) for k in ('CDF_grid','field_mean_m','field_std_m','relative_evidence')},minimum_ESS=min(r['moments']['ESS'] for r in high),maximum_weight=max(r['moments']['maximum_weight'] for r in high))

def run(p,protocol_record):
    out=frozen.fresh(ROOT/p['output']);ledger=dict(completed=False,protocol=protocol_record,pid=os.getpid(),cases=[],raw_local=True,TEST_opened=False,original_neural_results_unchanged=True);frozen.write(out/'summary.json',ledger);started=time.monotonic()
    def expired(*_):raise TimeoutError('Frozen900s CPU2 importance study cap; preserve partial cases')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,p['wall_cap_seconds'])
    try:
        for role in ('DEV','CAL'):
            public=frozen.load_arrays(p['roles'][role]['public']);params={s:frozen.load_arrays(rec) for s,rec in p['roles'][role]['parameters'].items()}
            for c in range(8):
                ledger['current_attempt']=dict(role=role,context=c,stage='reference_normalizer');frozen.write(out/'summary.json',ledger);ids,y,F,sides,refs=reference_group(p,public,role,c);frozen.write(out/f'{role}-context-{c}-reference-normalizer.json',refs)
                if any(r['new_reference_normalizer_relative_error']>p['reference']['relative_normalizer_tolerance'] for r in refs):raise RuntimeError('Resolution agreement of new evidence normalizer failed; attempted values preserved')
                for j,i in enumerate(ids):
                    directory=out/f'{role}-{i:03d}-context-{c}';directory.mkdir();case=dict(role=role,object_index=int(i),context=c,proposals={},reference_log_evidence=refs[j]['log_evidence'],new_reference_normalizer_relative_error=refs[j]['new_reference_normalizer_relative_error'],new_reference_group_seconds=refs[j]['new_reference_group_seconds']);ledger['cases'].append(case)
                    for proposal in ['prior']+list(map(str,frozen.FIT_SEEDS)):
                        pars=None if proposal=='prior' else frozen.slice_parameters(params[proposal],[int(i)*8+c]);records=[]
                        for scramble in SCRAMBLES:
                            ledger['current_attempt']=dict(role=role,context=c,object_index=int(i),proposal=proposal,scramble_seed=scramble,stage='weighted_cloud');frozen.write(out/'summary.json',ledger);frozen.guard_resources(p);seed=scramble+(0 if role=='DEV' else 10000)+100*int(i);u=sample_proposal(sobol_points(seed),pars);lw,q=evaluate_cloud(u,y[j],F,sides,pars,chunk=p['chunk']);value=dict(proposal=proposal,scramble_seed=seed,completed=True,budgets=[])
                            for N in BUDGETS:
                                s=summarize_cloud(u[:N],lw[:N],q[:N],p['grid']);value['budgets'].append(dict(count=N,moments=s,comparison=compare(s,refs[j])))
                            path=directory/f'{proposal}-scramble-{scramble}.json';frozen.write(path,value);records.append(value)
                        case['proposals'][proposal]=final_case_summary(records);case['proposals'][proposal]['records']=[frozen.record(directory/f'{proposal}-scramble-{s}.json') for s in SCRAMBLES];frozen.write(out/'summary.json',ledger)
                    print(role,int(i),c,'complete',round(time.monotonic()-started,2),flush=True)
        ledger.update(completed=True,all_case_proposals_numerically_passed=all(r['passed'] for c in ledger['cases'] for r in c['proposals'].values()),seconds=time.monotonic()-started,resources=frozen.guard_resources(p));frozen.write(out/'summary.json',ledger)
    except BaseException as e:
        ledger.update(seconds=time.monotonic()-started,failure=dict(type=type(e).__name__,message=str(e)));frozen.write(out/'summary.json',ledger);raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');args=a.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output));return
    p=validate(args.protocol,args.protocol_sha256);run(p,frozen.record(args.protocol))
if __name__=='__main__':main()
