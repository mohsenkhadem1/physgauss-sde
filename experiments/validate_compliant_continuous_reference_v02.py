"""Reporting-corrected continuation, reusing hash-bound completed adaptive refs.

v01 is unchanged. Its first completed SMC array/moments are restored, not
redrawn. All remaining original fixed seeds/counts/criteria/budgets retained.
"""
from __future__ import annotations
import argparse,hashlib,json,time,signal
from pathlib import Path
import numpy as np
import torch
from experiments import run_compliant_continuous_reference as original_driver
from experiments import campaign_compliant_continuous_reference as model

ROOT=Path(__file__).resolve().parents[1]
SEEDS=original_driver.SEEDS
COUNTS=original_driver.COUNTS

def scalar_safe(value):
 if isinstance(value,dict):return {k:scalar_safe(v) for k,v in value.items() if not k.startswith('_')}
 if isinstance(value,(list,tuple)):return [scalar_safe(v) for v in value]
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 return value

def write(path,value):Path(path).write_text(json.dumps(scalar_safe(value),indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rec(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))
def check(record):
 p=ROOT/record['path']
 if sha(p)!=record['sha256']:raise ValueError('Continuation pin changed: '+record['path'])
 return p

def save_failure(path,error,**metadata):
 write(path,dict(completed=False,error_type=type(error).__name__,error=str(error),**metadata))

def prepare(path):
 path=Path(path)
 if path.exists():raise ValueError('Fresh continuation protocol required')
 old=ROOT/'runs/operator-dev/compliant-continuous-reference-v01/protocol.json'
 protocol=json.loads(old.read_text());root=ROOT/'runs/gaussian-campaign-v01/compliant-continuous-reference-v01/conditional2D'
 paths=[root/'summary.json',root/'adaptive0-moments.npz',root/'adaptive1-moments.npz',
  root/f'smc-512-{SEEDS[0]}.npz',root/f'smc-512-{SEEDS[0]}-moments.npz']
 sources={**protocol['source_sha256'],**{p:sha(ROOT/p) for p in ['experiments/validate_compliant_continuous_reference_v02.py','tests/test_validate_compliant_continuous_reference_v02.py']}}
 value=dict(schema='compliant-continuous-validation-continuation-v0.2',status='FROZEN_BEFORE_REMAINING_SAMPLER_CALLS',
  original_protocol=rec(old),original_failed_terminal=rec(ROOT/'runs/operator-dev/compliant-continuous-reference-v01/failed-terminal.json'),
  retained_inputs=[rec(p) for p in paths],source_sha256=sources,
  same_prior_noise_observation_control_seeds_counts_criteria=True,CPU_threads=2,phase_caps_s=protocol['phase_caps_s'],
  completed_reference_reuse='Both hash-bound adaptive references and moments; zero new quadrature calls',
  completed_first_SMC_reuse='Restore first512/2026121101 particle+loglikelihood and A/B moments; no redraw. Its stage-level genealogy metadata was not saved by v01.',
  failure_policy='Preserve terminal metadata/partial outputs, no retries or seed selection',
  full4D_gate='All original2D reference, analytic, eachseed/count, MCMC and stability criteria must pass',
  versions=protocol['versions'])
 path.parent.mkdir(parents=True,exist_ok=True);write(path,value);return rec(path)


def restore_references(protocol):
 paths={Path(r['path']).name:check(r) for r in protocol['retained_inputs']}
 saved=json.loads(paths['summary.json'].read_text());refs=saved['adaptive_reference']
 if len(refs)!=2:raise ValueError('Exactly two completed adaptive references required')
 for i,ref in enumerate(refs):
  with np.load(paths[f'adaptive{i}-moments.npz'],allow_pickle=False) as data:
   ref['_mean']={p:data[p+'_mean'].copy() for p in ('A','B')};ref['_std']={p:data[p+'_std'].copy() for p in ('A','B')}
 delta=original_driver.comparison(refs[0],refs[1]);fields=delta['fields']
 agreement=dict(max_CDF_difference=delta['max_CDF_difference'],max_unit_quantile_difference=delta['max_unit_quantile_difference'],fields=fields,
  passed=delta['max_CDF_difference']<=2e-4 and delta['max_unit_quantile_difference']<=1e-4 and all(max(x['mean_RMS_m'],x['std_RMS_m'])<=1e-7 for x in fields.values()))
 return refs,agreement,saved['analytic_limit'],paths


def restored_summary(unit,dims,path):
 theta=model.physical(unit,dims);means={};stds={};targets={}
 with np.load(path,allow_pickle=False) as data:
  for p,start in [('A',25),('B',0)]:
   means[p]=data[p+'_mean'].copy();stds[p]=data[p+'_std'].copy()
   targets[p]=dict(latent_std_RMS_m=float(np.sqrt(np.mean(stds[p][start:]**2))))
 return dict(unit_CDF=[model.empirical_cdf(unit[:,j],original_driver.KNOTS).tolist() for j in range(unit.shape[1])],
  unit_quantiles=np.quantile(unit,[.05,.5,.95],axis=0).T.tolist(),
  physical_parameters=model.original.parameter_summary(theta,np.full(len(unit),1/len(unit))),targets=targets,_mean=means,_std=stds)


def phase(protocol,out,obs,forward,dimension,refs=None,agreement=None,analytic=None,paths=None):
 name='conditional2D' if dimension==2 else 'full4D';p=out/name;p.mkdir();start=time.monotonic()
 state=dict(phase=name,completed=False,passed=False,events=[],cap_s=protocol['phase_caps_s'][name])
 if dimension==2:state.update(adaptive_reference=refs,adaptive_tolerance_agreement=agreement,analytic_limit=analytic)
 def save():write(p/'summary.json',state)
 def alarm(_s,_f):raise original_driver.BudgetReached('Declared continuation phase wall cap reached')
 old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,state['cap_s'])
 try:
  dims=(0,3) if dimension==2 else (0,1,2,3);runs=[];pilot=None
  for count in COUNTS:
   for seed in SEEDS:
    label=f'smc-{count}-{seed}'
    restored=dimension==2 and count==512 and seed==SEEDS[0]
    if restored:
     with np.load(paths[label+'.npz'],allow_pickle=False) as data:
      unit=data['unit'].copy();ll=data['log_likelihood'].copy()
     if unit.shape!=(512,2) or ll.shape!=(512,) or not np.isfinite(unit).all() or not np.isfinite(ll).all():raise ValueError('Retained first SMC incomplete')
     summary=restored_summary(unit,dims,paths[label+'-moments.npz'])
     diagnostics=dict(seed=seed,particles=count,completed_particle_and_likelihood_arrays_restored=True,
      completed_moments_restored=True,stage_genealogy_diagnostics_not_saved_by_v01=True,no_new_physical_calls=True)
    else:
     result=model.smc(forward,dims,obs,count,seed);unit=result.state.coordinates;ll=result.log_likelihood;diagnostics=result.diagnostics()
     summary=original_driver.summaries(forward,unit,dims,p,label)
    np.savez_compressed(p/(label+'.npz'),unit=unit,log_likelihood=ll)
    if restored:
     np.savez_compressed(p/(label+'-moments.npz'),**{k+'_'+j:v for j,d in [('mean',summary['_mean']),('std',summary['_std'])] for k,v in d.items()})
    runs.append(dict(particles=count,seed=seed,SMC=diagnostics,summary=summary));pilot=unit
    state['SMC_runs']=runs;state['events'].append(label+(' restored' if restored else ' completed'));save()
  chain,diagnostics=model.continuous_mcmc(lambda x:forward.log_likelihood(x,dims,obs),pilot,
   seed=2026121110 if dimension==2 else 2026121120,warmup=500 if dimension==2 else 1000,draws=2000 if dimension==2 else 4000,
   progress=lambda step,total:write(p/'progress.json',dict(step=step,planned=total,elapsed_s=time.monotonic()-start)))
  np.savez_compressed(p/'MCMC.npz',unit=chain);mcmc=original_driver.summaries(forward,chain.reshape(-1,dimension),dims,p,'MCMC')
  state['MCMC']=dict(**diagnostics,summary=mcmc)
  state['SMC_pairwise_comparisons']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**original_driver.comparison(a['summary'],b['summary'])) for i,a in enumerate(runs) for b in runs[i+1:]]
  state['MCMC_vs_SMC']=[dict(particles=r['particles'],seed=r['seed'],**original_driver.comparison(r['summary'],mcmc)) for r in runs]
  if dimension==2:state['vs_adaptive_reference']=[dict(kind='SMC',particles=r['particles'],seed=r['seed'],**original_driver.comparison(r['summary'],refs[-1])) for r in runs]+[dict(kind='MCMC',**original_driver.comparison(mcmc,refs[-1]))]
  checks=[v['passed'] for v in state['SMC_pairwise_comparisons']+state['MCMC_vs_SMC']]+[diagnostics['diagnostics']['all_monitored_quantities_pass']]
  if dimension==2:checks += [agreement['passed'],analytic['passed']]+[v['passed'] for v in state['vs_adaptive_reference']]
  state.update(completed=True,passed=all(checks),status='COMPLETED_ALL_DECLARED_CHECKS')
 except Exception as exc:
  state.update(status='DECLARED_WALL_CAP' if isinstance(exc,original_driver.BudgetReached) else 'PRESERVED_FAILURE',error_type=type(exc).__name__,error=str(exc))
  save_failure(p/'failure.json',exc,elapsed_s=time.monotonic()-start,events=state['events'],no_retry=True)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);state['elapsed_s']=time.monotonic()-start;save()
 return state


def run(protocol_path,digest,output):
 protocol=json.loads(check(dict(path=protocol_path,sha256=digest)).read_text())
 old=json.loads(check(protocol['original_protocol']).read_text())
 for path,h in protocol['source_sha256'].items():check(dict(path=path,sha256=h))
 check(protocol['original_failed_terminal']);refs,agreement,analytic,paths=restore_references(protocol)
 if protocol['phase_caps_s']!={'conditional2D':180,'full4D':360}:raise ValueError('Unchanged phase caps required')
 out=Path(output)
 if out.exists():raise ValueError('Fresh continuation output required')
 out.mkdir(parents=True);write(out/'protocol.json',protocol);torch.set_num_threads(2)
 with np.load(check(old['observation_archive']),allow_pickle=False) as data:obs=data['observation'].copy()
 forward=model.ExactVectorForward();two=phase(protocol,out,obs,forward,2,refs,agreement,analytic,paths)
 result=dict(schema='manufactured-continuous-validation-result-v0.2',protocol=dict(path=protocol_path,sha256=digest),conditional2D=two,
  full4D_started=bool(two['passed']),physical_calibration_claim=False,continuous_accuracy_proven=False,old_grid_and_reporting_failures_preserved=True,raw_arrays_LOCAL=True)
 if two['passed']:result['full4D']=phase(protocol,out,obs,forward,4)
 else:result['full4D_not_started_reason']='Original2D validation gate did not pass; no4D claim'
 result['numerical_validation_passed']=bool(two['passed'] and result.get('full4D',{}).get('passed',False));write(out/'summary.json',result)
 return dict(conditional2D_completed=two['completed'],conditional2D_passed=two['passed'],full4D_started=result['full4D_started'],numerical_validation_passed=result['numerical_validation_passed'])

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for name in ('protocol','protocol-sha256','output'):p.add_argument('--'+name,required=True)
 a=parser.parse_args();print(json.dumps(prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)))
