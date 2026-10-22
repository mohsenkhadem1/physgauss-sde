"""Complete ONLY two unfinished fixed SMC slots with parity-checked exact LL.

Four completed high-count runs, adaptive references and MCMC are hash-reused.
The original wall-cap and lower-resolution failures remain unchanged.
"""
from __future__ import annotations
import argparse,json,time,signal
from pathlib import Path
import numpy as np
import torch
from experiments import diagnose_compliant_smc_resolution_v03 as resolution
from experiments import validate_compliant_continuous_reference_v02 as metadata
from experiments import campaign_compliant_observation_forward as observation

ROOT=metadata.ROOT
REMAINING=(2026121102,2026121103)
CAP=180

def prepare(path):
 path=Path(path)
 if path.exists():raise ValueError('Fresh completion protocol required')
 protocol=ROOT/'runs/operator-dev/compliant-smc-resolution-v03/protocol.json';old=json.loads(protocol.read_text())
 summary=ROOT/'runs/gaussian-campaign-v01/compliant-smc-resolution-v03/summary.json';d=json.loads(summary.read_text())
 expected=[(8192,s) for s in resolution.first.SEEDS]+[(32768,2026121101)]
 if [(r['particles'],r['seed']) for r in d['runs']]!=expected or d['status']!='DECLARED_WALL_CAP':raise ValueError('Exact four completed slots + preserved cap required')
 pins=[metadata.rec(summary)]
 for count,seed in expected:
  for suffix in ('.npz','-moments.npz'):pins.append(metadata.rec(summary.parent/f'smc-{count}-{seed}{suffix}'))
 sources={**old['source_sha256'],**{p:metadata.sha(ROOT/p) for p in ['experiments/complete_compliant_smc_resolution_v04.py',
  'experiments/campaign_compliant_observation_forward.py','tests/test_campaign_compliant_observation_forward.py']}}
 value=dict(schema='compliant-SMC-resolution-completion-v0.4',status='FROZEN_BEFORE_COMPLETION_CALLS',prior_protocol=metadata.rec(protocol),
  retained_high_count=pins,source_sha256=sources,remaining=[dict(particles=32768,seed=s) for s in REMAINING],CPU_threads=2,wall_cap_s=CAP,
  exact_helper='Float64 same held-force mechanics; frames12/24,nodes1/4,stop24; fixed1024chunks; fullforward and likelihood parity2testsPASS1.70s',
  interrupted_seed='32768/2026121102 restarted from SAME deterministicseed because v03 wallcap produced no complete particlearray; no redraw or selection',
  same_physics_observation_prior_noise_algorithms_criteria=True,reference_MCMC_four_completed_SMC_not_rerun=True,
  no_hidden_retry_or_dropped_results=True,no4D=True,scope='Conditional2D manufactured numerical reference, NOT physical calibration')
 path.parent.mkdir(parents=True,exist_ok=True);metadata.write(path,value);return metadata.rec(path)


def run(path,digest,output):
 plan=json.loads(metadata.check(dict(path=path,sha256=digest)).read_text())
 for p,h in plan['source_sha256'].items():metadata.check(dict(path=p,sha256=h))
 if plan['remaining']!=[dict(particles=32768,seed=s) for s in REMAINING] or plan['wall_cap_s']!=CAP:raise ValueError('Exact remaining2 fixedslots required')
 oldplan=json.loads(metadata.check(plan['prior_protocol']).read_text());v02=json.loads(metadata.check(oldplan['prior_protocol']).read_text())
 refs,agreement,analytic,_=metadata.restore_references(v02)
 original=json.loads(metadata.check(v02['original_protocol']).read_text())
 with np.load(metadata.check(original['observation_archive']),allow_pickle=False) as raw:obs=raw['observation'].copy()
 oldfiles={r['path']:metadata.check(r) for r in oldplan['retained_inputs']}
 mrecord=next(r for r in oldplan['retained_inputs'] if r['path'].endswith('compliant-continuous-reference-v02/conditional2D/summary.json'))
 msource=json.loads(metadata.check(mrecord).read_text());mcmc=msource['MCMC']['summary']
 moment=next(p for name,p in oldfiles.items() if name.endswith('MCMC-moments.npz'))
 with np.load(moment,allow_pickle=False) as raw:
  mcmc['_mean']={p:raw[p+'_mean'].copy() for p in ('A','B')};mcmc['_std']={p:raw[p+'_std'].copy() for p in ('A','B')}
 files={Path(r['path']).name:metadata.check(r) for r in plan['retained_high_count']}
 old=json.loads(files['summary.json'].read_text());runs=old['runs']
 for row in runs:
  label=f"smc-{row['particles']}-{row['seed']}"
  with np.load(files[label+'-moments.npz'],allow_pickle=False) as raw:
   row['summary']['_mean']={p:raw[p+'_mean'].copy() for p in ('A','B')};row['summary']['_std']={p:raw[p+'_std'].copy() for p in ('A','B')}
  row['restored_unchanged_from_v03']=True
 out=Path(output)
 if out.exists():raise ValueError('Fresh completion output required')
 out.mkdir(parents=True);metadata.write(out/'protocol.json',plan);torch.set_num_threads(2);forward=observation.ObservationOnlyForward()
 start=time.monotonic();state=dict(schema='compliant-SMC-resolution-completed-result-v0.4',completed=False,passed=False,runs=runs,
  adaptive_tolerance_agreement=agreement,analytic_limit=analytic,MCMC_diagnostics=msource['MCMC']['diagnostics'],
  all_lower_count_reporting_and_wallcap_failures_preserved=True,physical_calibration_claim=False,continuous_accuracy_proven=False,raw_arrays_LOCAL=True,full4D_started=False)
 def save():metadata.write(out/'summary.json',state)
 def alarm(_s,_f):raise resolution.first.BudgetReached('Completion180s cap reached')
 handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,CAP)
 try:
  for seed in REMAINING:
   result=resolution.model.smc(forward,(0,3),obs,32768,seed);label=f'smc-32768-{seed}'
   np.savez_compressed(out/(label+'.npz'),unit=result.state.coordinates,log_likelihood=result.log_likelihood)
   summary=resolution.first.summaries(forward,result.state.coordinates,(0,3),out,label)
   state['runs'].append(dict(particles=32768,seed=seed,SMC=result.diagnostics(),summary=summary,
    vs_adaptive=resolution.first.comparison(summary,refs[-1]),vs_MCMC=resolution.first.comparison(summary,mcmc)))
   save()
  state['pairwise']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**resolution.first.comparison(a['summary'],b['summary'])) for i,a in enumerate(runs) for b in runs[i+1:]]
  passed=agreement['passed'] and analytic['passed'] and msource['MCMC']['diagnostics']['all_monitored_quantities_pass']
  passed=passed and all(r['vs_adaptive']['passed'] and r['vs_MCMC']['passed'] for r in runs) and all(r['passed'] for r in state['pairwise'])
  state.update(completed=True,passed=bool(passed),status='ALL_SIX_FIXED_SLOTS_COMPLETED')
 except Exception as exc:
  state.update(status='DECLARED_WALL_CAP' if isinstance(exc,resolution.first.BudgetReached) else 'PRESERVED_FAILURE',error_type=type(exc).__name__,error=str(exc))
  metadata.save_failure(out/'failure.json',exc,completed_slots=len(runs),no_retry=True)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler);state['elapsed_s']=time.monotonic()-start;save()
 return {k:state[k] for k in ('completed','passed','status','elapsed_s')}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for name in ('protocol','protocol-sha256','output'):p.add_argument('--'+name,required=True)
 a=parser.parse_args();print(json.dumps(prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)))
