"""Fixed high-count SMC resolution diagnosis; all earlier failures immutable."""
from __future__ import annotations
import argparse,json,time,signal
from pathlib import Path
import numpy as np
import torch
from experiments import validate_compliant_continuous_reference_v02 as previous
from experiments import run_compliant_continuous_reference as first
from experiments import campaign_compliant_continuous_reference as model

ROOT=previous.ROOT
COUNTS=(8192,32768)
CAP_S=180

def prepare(path):
 path=Path(path)
 if path.exists():raise ValueError('Fresh high-count protocol required')
 old=ROOT/'runs/operator-dev/compliant-continuous-reference-v02/protocol.json';v02=json.loads(old.read_text())
 retained=list(v02['retained_inputs'])+[previous.rec(ROOT/'runs/gaussian-campaign-v01/compliant-continuous-reference-v02/conditional2D/summary.json'),
  previous.rec(ROOT/'runs/gaussian-campaign-v01/compliant-continuous-reference-v02/conditional2D/MCMC-moments.npz')]
 source={**v02['source_sha256'],'experiments/diagnose_compliant_smc_resolution_v03.py':previous.sha(ROOT/'experiments/diagnose_compliant_smc_resolution_v03.py')}
 value=dict(schema='compliant-SMC-resolution-v0.3',status='FROZEN_BEFORE_HIGH_COUNT_CALLS',prior_protocol=previous.rec(old),
  retained_inputs=retained,source_sha256=source,counts=list(COUNTS),seeds=list(first.SEEDS),CPU_threads=2,wall_cap_s=CAP_S,
  same_exact_forward_observation_prior_noise_controls=True,same_CDF_quantile_forecast_criteria=True,
  reference_reuse='Independent two-tolerance adaptive2D + eight-prior-started-chain MCMC; no quadrature or MCMC reruns',
  all_lower_count_and_serialization_failures_preserved=True,
  qualification='Every new seed/count vs adaptive and MCMC plus all15pair comparisons must pass original criteria',
  no_4D_calls_in_this_diagnosis=True,no_retry_seed_or_outcome_selection=True,scope='Conditional2D numerical resolution only, not physical calibration')
 path.parent.mkdir(parents=True,exist_ok=True);previous.write(path,value);return previous.rec(path)


def run(path,digest,output):
 protocol=json.loads(previous.check(dict(path=path,sha256=digest)).read_text())
 for p,h in protocol['source_sha256'].items():previous.check(dict(path=p,sha256=h))
 if protocol['counts']!=list(COUNTS) or protocol['seeds']!=list(first.SEEDS) or protocol['wall_cap_s']!=CAP_S:raise ValueError('Fixed high-count plan required')
 v02=json.loads(previous.check(protocol['prior_protocol']).read_text());refs,agreement,analytic,_=previous.restore_references(v02)
 files={Path(r['path']).name:previous.check(r) for r in protocol['retained_inputs']}
 # Both retained summary files share a basename; explicitly bind v02 by path.
 summary_record=next(r for r in protocol['retained_inputs'] if r['path'].endswith('compliant-continuous-reference-v02/conditional2D/summary.json'))
 old=json.loads(previous.check(summary_record).read_text());mcmc=old['MCMC']['summary']
 with np.load(files['MCMC-moments.npz'],allow_pickle=False) as data:
  mcmc['_mean']={p:data[p+'_mean'].copy() for p in ('A','B')};mcmc['_std']={p:data[p+'_std'].copy() for p in ('A','B')}
 original=json.loads(previous.check(v02['original_protocol']).read_text())
 with np.load(previous.check(original['observation_archive']),allow_pickle=False) as data:obs=data['observation'].copy()
 out=Path(output)
 if out.exists():raise ValueError('Fresh high-count output required')
 out.mkdir(parents=True);previous.write(out/'protocol.json',protocol);torch.set_num_threads(2);forward=model.ExactVectorForward()
 start=time.monotonic();state=dict(schema='compliant-SMC-resolution-result-v0.3',completed=False,passed=False,runs=[],
  adaptive_tolerance_agreement=agreement,analytic_limit=analytic,MCMC_diagnostics=old['MCMC']['diagnostics'],
  lower_count_gate_failure_preserved=True,CPU_threads=2,wall_cap_s=CAP_S,raw_arrays_LOCAL=True,full4D_started=False,
  physical_calibration_claim=False,continuous_accuracy_proven=False)
 def save():previous.write(out/'summary.json',state)
 def alarm(_s,_f):raise first.BudgetReached('High-count180s wall cap reached')
 handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,CAP_S)
 try:
  for count in COUNTS:
   for seed in first.SEEDS:
    result=model.smc(forward,(0,3),obs,count,seed);label=f'smc-{count}-{seed}'
    np.savez_compressed(out/(label+'.npz'),unit=result.state.coordinates,log_likelihood=result.log_likelihood)
    summary=first.summaries(forward,result.state.coordinates,(0,3),out,label)
    state['runs'].append(dict(particles=count,seed=seed,SMC=result.diagnostics(),summary=summary,
     vs_adaptive=first.comparison(summary,refs[-1]),vs_MCMC=first.comparison(summary,mcmc)))
    save()
  state['pairwise']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**first.comparison(a['summary'],b['summary'])) for i,a in enumerate(state['runs']) for b in state['runs'][i+1:]]
  passed=agreement['passed'] and analytic['passed'] and old['MCMC']['diagnostics']['all_monitored_quantities_pass']
  passed=passed and all(x['vs_adaptive']['passed'] and x['vs_MCMC']['passed'] for x in state['runs']) and all(x['passed'] for x in state['pairwise'])
  state.update(completed=True,passed=bool(passed),status='ALL_FIXED_RUNS_COMPLETED')
 except Exception as exc:
  state.update(status='DECLARED_WALL_CAP' if isinstance(exc,first.BudgetReached) else 'PRESERVED_FAILURE',error_type=type(exc).__name__,error=str(exc))
  previous.save_failure(out/'failure.json',exc,completed_runs=len(state['runs']),no_retry=True)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler);state['elapsed_s']=time.monotonic()-start;save()
 return {k:state[k] for k in ('completed','passed','status','elapsed_s')}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for name in ('protocol','protocol-sha256','output'):p.add_argument('--'+name,required=True)
 a=parser.parse_args();print(json.dumps(prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)))
